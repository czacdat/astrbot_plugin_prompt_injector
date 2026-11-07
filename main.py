# -*- coding: utf-8 -*-
"""生图提示词注入插件。

思路：不改 astrbot_plugin_image_generation 的任何源码，也不经过大模型，
在 **工具调用层** 用猴子补丁拦住 generate_image，把它 kwargs 里的 prompt
在最前面拼上自定义的固定前缀（画师词、质量词之类）。

工作方式：
  1. 找到工具管理器里 name == target_tool（默认 generate_image）的工具；
  2. 给它所属的类打补丁，替换 cls.call；
  3. 每次调用先改 kwargs["prompt"]，再交给原来的 call 执行。

因为生图插件可能比本插件后加载，所以初始化后会用后台任务每 5 秒检查一次，
发现工具出现（或插件被重载导致补丁丢失）就重新打上。

配置持久化在 AstrBot 的插件配置里（面板可直接改），
聊天命令 /注入前缀 也能改，改完立刻写回配置文件，重启不丢。
"""

from __future__ import annotations

import asyncio
import inspect
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:  # 用来判断「目标是自定义 call 还是走 handler」
    from astrbot.core.agent.tool import FunctionTool as _FUNC_TOOL_BASE
except Exception:  # noqa: BLE001 - 导入失败时退化成按类名判断
    _FUNC_TOOL_BASE = None

PLUGIN_NAME = "astrbot_plugin_prompt_injector"
LOG = "[PromptInjector]"

# 补丁标记：挂在被补丁的类上，用来判断是否已经打过
PATCH_FLAG = "_prompt_injector_patched"
PATCH_ORIGINAL = "_prompt_injector_original"
PATCH_OWNER = "_prompt_injector_owner"
PATCH_WRAPPER = "_prompt_injector_wrapper"


def _is_base_call(owner: type | None) -> bool:
    """owner 是不是框架基类 FunctionTool（基类上的 call 不是「自定义实现」）。"""
    if owner is None:
        return True
    if _FUNC_TOOL_BASE is not None:
        return owner is _FUNC_TOOL_BASE
    return getattr(owner, "__name__", "") == "FunctionTool"


@register(
    PLUGIN_NAME,
    "DSH (AI)",
    "生图提示词注入：在工具调用层给 generate_image 的 prompt 加固定前缀，不改生图插件源码",
    "1.0.2",
)
class PromptInjectorPlugin(Star):
    """在工具调用层给生图 prompt 加固定前缀。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        # 注意：AstrBot 4.28 的 Star 基类不会自动保存 config，必须自己留一份，
        # 否则会报 AttributeError: 'xxx' object has no attribute 'config'
        self.config = config if config is not None else {}
        self._ensure_task: asyncio.Task | None = None
        self._warned: dict[str, bool] = {}
        self._stopped = False
        # 记录本实例打过补丁的类；切换 target_tool 时用来回收旧补丁
        self._patched_classes: list[type] = []
        # handler 型工具（@filter.llm_tool）：按实例打补丁，不碰共享基类
        self._handler_patches: list[tuple[Any, Any]] = []
        self._reload_cfg()

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #
    def _reload_cfg(self) -> None:
        """把面板/命令里的配置读进内存（每次改动后都要调一次）。"""
        cfg = dict(getattr(self, "config", None) or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.prefix = str(cfg.get("prefix") or "").strip()
        self.separator = str(cfg.get("separator", ", "))
        if not self.separator:
            self.separator = ", "
        self.target_tool = str(cfg.get("target_tool") or "generate_image").strip() or "generate_image"
        self.dedupe = bool(cfg.get("dedupe", True))
        self.log_injections = bool(cfg.get("log_injections", True))
        raw_wl = cfg.get("whitelist_groups") or []
        if isinstance(raw_wl, str):
            raw_wl = re.split(r"[,，\s]+", raw_wl)
        self.whitelist_groups = [str(x).strip() for x in raw_wl if str(x).strip()]

    async def _persist(self, **updates: Any) -> bool:
        """把改动写进 AstrBot 插件配置，重启后仍生效。"""
        try:
            for key, value in updates.items():
                self.config[key] = value
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{LOG} 写入配置失败: {exc}")
            return False
        try:
            saver = getattr(self.config, "save_config_async", None)
            if saver is not None:
                await saver()
            else:
                self.config.save_config()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{LOG} 保存配置失败: {exc}")
            return False
        self._reload_cfg()
        return True

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    async def initialize(self) -> None:
        """加载时就试一次，之后每 5 秒补一次（应对生图插件后加载/被重载）。"""
        self._stopped = False
        # initialize 被重复调用时先收掉上一条循环，避免两条自检任务并存
        if self._ensure_task is not None and not self._ensure_task.done():
            self._ensure_task.cancel()
            try:
                await self._ensure_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._ensure_task = None
        self._reload_cfg()
        self._try_patch()
        self._ensure_task = asyncio.create_task(self._ensure_loop())

    async def terminate(self) -> None:
        """插件卸载/停用时调用：停掉自检循环，并把补丁全部还原。

        必须还原，否则停用插件后 patched_call 还挂在类上，
        注入会继续生效，而且它闭包里持有本实例，导致无法回收。
        """
        self._stopped = True
        task, self._ensure_task = self._ensure_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"{LOG} 自检任务结束异常: {exc}")
        self._restore_all_patches()

    def _restore_all_patches(self) -> None:
        """还原本实例打过的所有补丁（类补丁 + handler 补丁）。"""
        for cls in list(self._patched_classes):
            if self._release_patch_on(cls):
                logger.info(f"{LOG} 已还原 {cls.__name__} 的补丁")
        self._patched_classes.clear()
        for tool, original in list(self._handler_patches):
            try:
                tool.handler = original
                logger.info(
                    f"{LOG} 已还原工具 {getattr(tool, 'name', '?')} 的 handler"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{LOG} 还原 handler 失败: {exc}")
        self._handler_patches.clear()

    def _release_patch_on(self, cls: type) -> bool:
        """还原某个类上的补丁；只还原「本实例打的」，避免误撤别的实例的补丁。

        返回是否真的还原了。"""
        if cls.__dict__.get(PATCH_OWNER) is not self:
            return False
        wrapper = cls.__dict__.get(PATCH_WRAPPER)
        current = cls.__dict__.get("call")
        if wrapper is not None and current is not None and current is not wrapper:
            # 有人在我们上面又包了一层，硬还原会把别人的包装抹掉
            logger.warning(
                f"{LOG} {cls.__name__}.call 已被其他代码替换，跳过还原以免破坏"
            )
            return False
        original = cls.__dict__.get(PATCH_ORIGINAL)
        if original is not None:
            try:
                setattr(cls, "call", original)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{LOG} 还原 {cls.__name__} 的 call 失败: {exc}")
                return False
        for attr in (PATCH_FLAG, PATCH_ORIGINAL, PATCH_OWNER):
            try:
                delattr(cls, attr)
            except AttributeError:
                pass
        return True

    def _release_stale_patches(self, keep: type | None) -> None:
        """回收目标之外、由本实例打过的补丁。

        keep=None 表示「当前没有目标」——全部回收，避免旧目标在目标缺失期间继续被注入。
        """
        for cls in list(self._patched_classes):
            if keep is not None and cls is keep:
                continue
            if self._release_patch_on(cls):
                logger.info(
                    f"{LOG} 目标工具已切换或缺失，已还原 {cls.__name__} 的旧补丁"
                )
            self._patched_classes.remove(cls)
        # handler 补丁同理：目标没了就还原
        if keep is None:
            for tool, original in list(self._handler_patches):
                try:
                    tool.handler = original
                except Exception:  # noqa: BLE001
                    pass
                self._handler_patches.remove((tool, original))

    async def _ensure_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(5)
                self._try_patch()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{LOG} 补丁守护异常: {exc}")

    # ------------------------------------------------------------------ #
    # 打补丁
    # ------------------------------------------------------------------ #
    def _find_tool(self):
        """按框架语义找目标工具：最后一个同名且 active 的实例。"""
        try:
            manager = self.context.get_llm_tool_manager()
        except Exception as exc:  # noqa: BLE001
            self._maybe_warn("manager", f"拿不到工具管理器: {exc}")
            return None
        getter = getattr(manager, "get_func", None)
        if callable(getter):
            try:
                tool = getter(self.target_tool)
            except Exception as exc:  # noqa: BLE001
                self._maybe_warn("get_func", f"查询工具失败: {exc}")
                tool = None
            if tool is not None and not getattr(tool, "active", True):
                return None
            return tool
        # 兜底：自己按「后加载优先 + 只看 active」遍历
        for tool in reversed(getattr(manager, "func_list", []) or []):
            if getattr(tool, "name", "") == self.target_tool and getattr(
                tool, "active", True
            ):
                return tool
        return None

    def _maybe_warn(self, reason: str, message: str) -> None:
        """同一个原因只提醒一次，避免每 5 秒刷屏。"""
        if not self._warned.get(reason):
            logger.info(f"{LOG} {message}（之后不再重复提示）")
            self._warned[reason] = True

    def _try_patch(self) -> bool:
        """给目标工具打补丁；已经打过就直接返回 True。"""
        if self._stopped:
            return False
        tool = self._find_tool()
        if tool is None:
            # H1：目标不存在时也要把旧补丁撤掉，否则旧工具会继续被注入
            self._release_stale_patches(None)
            self._maybe_warn(
                "missing_tool",
                f"还没找到工具 {self.target_tool}，等待目标插件加载",
            )
            return False

        cls = type(tool)
        owner = self._call_owner(cls)
        if _is_base_call(owner):
            # handler 型工具（@filter.llm_tool）：包实例的 handler，绝不碰共享基类
            self._release_stale_patches(None)
            return self._patch_handler(tool)

        # 目标可能被配置改过：先把旧目标上的补丁撤掉，避免两个类同时被注入
        self._release_stale_patches(cls)
        # 只看类自己身上的标记：用 getattr 会把父类的标记继承过来，导致子类被误判为「已打过」
        # 同时检查「打补丁的是不是当前这个插件实例」：插件被重载后，
        # 旧实例的包装函数还挂在类上，配置会变旧，必须用新实例重新包一次
        if cls.__dict__.get(PATCH_FLAG) and cls.__dict__.get(PATCH_OWNER) is self:
            if cls not in self._patched_classes:
                self._patched_classes.append(cls)
            return True

        # 始终基于「最初的原始实现」重新包装，避免重载后套娃
        original = cls.__dict__.get(PATCH_ORIGINAL) or getattr(cls, "call", None)
        if original is None:
            self._maybe_warn(
                "no_call", f"{cls.__name__} 没有 call 方法，也无法走 handler"
            )
            return False

        plugin = self

        async def patched_call(tool_self, context, *args, **kwargs):  # noqa: ANN001
            # 先改参数，再交给原来的实现；任何异常都不能影响生图
            try:
                kwargs = plugin._inject_kwargs(context, kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{LOG} 注入失败，按原样执行: {exc}")
            return await original(tool_self, context, *args, **kwargs)

        # 保留函数元信息（有些框架会读 __name__ / __signature__）
        try:
            patched_call.__name__ = getattr(original, "__name__", "call")
            patched_call.__qualname__ = getattr(original, "__qualname__", "call")
            patched_call.__doc__ = getattr(original, "__doc__", None)
            patched_call.__wrapped__ = original
            patched_call.__signature__ = inspect.signature(original)
        except Exception:  # noqa: BLE001
            pass

        setattr(cls, "call", patched_call)
        setattr(cls, PATCH_FLAG, True)
        setattr(cls, PATCH_ORIGINAL, original)
        setattr(cls, PATCH_OWNER, self)
        setattr(cls, PATCH_WRAPPER, patched_call)
        if cls not in self._patched_classes:
            self._patched_classes.append(cls)
        self._warned.pop("missing_tool", None)
        logger.info(
            f"{LOG} 已拦截工具 {self.target_tool}（类 {cls.__name__}，自定义 call）"
            f"，当前前缀: {self.prefix or '（空，不注入）'}"
        )
        return True

    def _call_owner(self, cls: type) -> type | None:
        """沿 MRO 找真正定义 call 的类（判断是自定义 call 还是框架基类）。"""
        for base in getattr(cls, "__mro__", [cls]):
            if "call" in getattr(base, "__dict__", {}):
                return base
        return None

    def _patch_handler(self, tool) -> bool:
        """handler 型工具：只替换这一个实例的 handler，不碰共享基类。

        AstrBot 的 _PermissionGuardedTool 对「有 handler 的工具」直接调用 handler，
        根本不会走 cls.call —— 所以对这类工具，类补丁是无效的。
        """
        for patched_tool, _original in self._handler_patches:
            if patched_tool is tool:
                return True
        original = getattr(tool, "handler", None)
        if original is None:
            self._maybe_warn(
                "no_handler",
                f"工具 {self.target_tool} 既没有自定义 call 也没有 handler，无法拦截",
            )
            return False

        plugin = self

        def patched_handler(event, **kwargs):
            # handler 可能是同步/异步/异步生成器，这里只改参数，原样转交
            try:
                kwargs = plugin._inject_kwargs_for_event(event, kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{LOG} 注入失败，按原样执行: {exc}")
            return original(event, **kwargs)

        try:
            tool.handler = patched_handler
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{LOG} 无法替换 {self.target_tool} 的 handler: {exc}")
            return False

        self._handler_patches.append((tool, original))
        self._warned.pop("missing_tool", None)
        logger.info(
            f"{LOG} 已拦截工具 {self.target_tool}（实例 handler）"
            f"，当前前缀: {self.prefix or '（空，不注入）'}"
        )
        return True

    # ------------------------------------------------------------------ #
    # 注入逻辑
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_event(context) -> Any:
        """从工具调用的 context 里取出消息事件（兼容几种包装）。"""
        wrapped = getattr(context, "context", None)
        event = getattr(wrapped, "event", None)
        if event is not None:
            return event
        event = getattr(context, "event", None)
        if event is not None:
            return event
        if isinstance(context, dict):
            return context.get("event")
        return None

    def _allowed(self, event) -> bool:
        """白名单判断：留空 = 全局生效；配了就只认严格匹配。"""
        if not self.whitelist_groups:
            return True
        if event is None:
            # 限制器应当 fail-closed：拿不到事件就不能确认是否在白名单内
            self._maybe_warn("no_event", "拿不到消息事件，白名单判定为不通过")
            return False
        try:
            gid = str(event.get_group_id() or "").strip()
        except Exception:  # noqa: BLE001
            gid = ""
        if gid:
            return gid in self.whitelist_groups
        # 私聊/无群号：只接受完整 umo 严格相等，避免用子串误命中
        try:
            umo = str(getattr(event, "unified_msg_origin", "") or "")
        except Exception:  # noqa: BLE001
            umo = ""
        return bool(umo) and umo in self.whitelist_groups

    def _command_visible(self, event) -> bool:
        """命令是否可见/可回应。

        白名单为空 → 所有人可见；
        配了白名单 → 只有白名单群能看到这个命令；
        私聊只给管理员用（避免把管理员自己锁在外面）。
        不可见时完全静默，不回复任何内容。
        """
        if not self.whitelist_groups:
            return True
        try:
            gid = str(event.get_group_id() or "").strip()
        except Exception:  # noqa: BLE001
            gid = ""
        if not gid:
            try:
                return bool(event.is_admin())
            except Exception:  # noqa: BLE001
                return False
        return gid in self.whitelist_groups

    @staticmethod
    def _norm_prompt(text: str) -> str:
        """规范化用于比较：忽略空格/中英文逗号差异，避免「只差一个空格」被判成没注入过。"""
        return re.sub(r"[\s,，、]+", ",", str(text or "").lower()).strip(",")

    def _apply_prefix(self, prompt: str) -> str:
        """把前缀拼到最前面；开启防重复时，开头已有就不再加。"""
        prefix = self.prefix.strip()
        if not prefix:
            return prompt
        body = prompt.strip()
        if self.dedupe and self._norm_prompt(body).startswith(self._norm_prompt(prefix)):
            return prompt
        if not body:
            return prefix
        return f"{prefix}{self.separator}{body}"

    def _inject_kwargs(self, context, kwargs: dict) -> dict:
        """自定义 call 型工具：从 context 里取事件后注入。"""
        return self._inject_kwargs_for_event(self._extract_event(context), kwargs)

    def _inject_kwargs_for_event(self, event, kwargs: dict) -> dict:
        """核心：改 kwargs 里的 prompt。"""
        # 每次注入前重读一次配置：这样面板里改开关/前缀能立刻生效，不用重载插件
        try:
            self._reload_cfg()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"{LOG} 重读配置失败，沿用上次的值: {exc}")
        if not self.enabled or not self.prefix:
            return kwargs
        if not isinstance(kwargs, dict):
            return kwargs
        prompt = kwargs.get("prompt")
        if prompt is None:
            prompt = ""
        if not isinstance(prompt, str):
            return kwargs

        if not self._allowed(event):
            return kwargs

        new_prompt = self._apply_prefix(prompt)
        if new_prompt == prompt:
            return kwargs

        kwargs = dict(kwargs)
        kwargs["prompt"] = new_prompt
        if self.log_injections:
            gid = ""
            try:
                gid = str(event.get_group_id() or "") if event is not None else ""
            except Exception:  # noqa: BLE001
                gid = ""
            # 默认只记长度：prompt 里可能有群内容/敏感词，不该整段落进日志
            logger.info(
                f"{LOG} 已注入前缀（{'群' + gid if gid else '私聊'}）: "
                f"+{len(new_prompt) - len(prompt)} 字符，共 {len(new_prompt)} 字符"
            )
            logger.debug(f"{LOG} 注入后 prompt 前 120 字: {new_prompt[:120]}")
        return kwargs

    # ------------------------------------------------------------------ #
    # 聊天命令
    # ------------------------------------------------------------------ #
    def _status_text(self) -> str:
        wl = "、".join(self.whitelist_groups) if self.whitelist_groups else "不限（所有群和私聊）"
        return (
            "📝 生图提示词注入\n"
            f"状态：{'✅ 开启' if self.enabled else '⛔ 关闭'}\n"
            f"前缀：{self.prefix or '（空，等于不注入）'}\n"
            f"目标工具：{self.target_tool}\n"
            f"防重复：{'开' if self.dedupe else '关'}\n"
            f"生效范围：{wl}"
        )

    @filter.command("注入前缀")
    async def inject_prefix_command(self, event: AstrMessageEvent):
        """生图提示词注入开关与设置。"""
        event.stop_event()

        # 不在白名单的会话：完全静默，连「有这个插件」都不暴露
        if not self._command_visible(event):
            return

        # 解析「/注入前缀 设置 xxx」里的完整剩余文本
        raw = (event.message_str or "").strip()
        body = re.sub(r"^[/!！]?(注入前缀)\s*", "", raw, count=1).strip()
        part = body.split(maxsplit=1)
        action = part[0] if part else ""
        value = part[1].strip() if len(part) > 1 else ""

        # 纯查询不需要管理员
        if not action or action in ("状态", "status"):
            yield event.plain_result(self._status_text())
            return

        if not event.is_admin():
            yield event.plain_result("❌ 只有管理员能改注入设置。想查状态可以直接 /注入前缀")
            return

        if action in ("开启", "开", "on", "启用"):
            ok = await self._persist(enabled=True)
            yield event.plain_result(
                ("✅ 已开启注入\n前缀：" + (self.prefix or "（空，请用 /注入前缀 设置 <内容> 填写）"))
                if ok
                else "❌ 保存失败，请看插件日志"
            )
            return

        if action in ("关闭", "关", "off", "禁用"):
            ok = await self._persist(enabled=False)
            yield event.plain_result("✅ 已关闭注入（配置保留，随时可再开）" if ok else "❌ 保存失败，请看插件日志")
            return

        if action in ("设置", "set"):
            if not value:
                yield event.plain_result("用法：/注入前缀 设置 masterpiece, best quality")
                return
            ok = await self._persist(prefix=value, enabled=True)
            yield event.plain_result(
                f"✅ 前缀已更新并开启：\n{value}" if ok else "❌ 保存失败，请看插件日志"
            )
            return

        if action in ("白名单", "add", "添加白名单"):
            items = [x for x in re.split(r"[,，\s]+", value) if x]
            if not items:
                yield event.plain_result("用法：/注入前缀 白名单 群号1 群号2")
                return
            merged = list(dict.fromkeys(self.whitelist_groups + items))
            ok = await self._persist(whitelist_groups=merged)
            if not ok:
                yield event.plain_result("❌ 保存失败，请看插件日志")
                return
            yield event.plain_result("✅ 白名单已更新：\n" + "、".join(merged))
            return

        if action in ("移除白名单", "del", "删除白名单"):
            items = [x for x in re.split(r"[,，\s]+", value) if x]
            if not items:
                yield event.plain_result("用法：/注入前缀 移除白名单 群号")
                return
            merged = [g for g in self.whitelist_groups if g not in items]
            ok = await self._persist(whitelist_groups=merged)
            yield event.plain_result(
                ("✅ 已移除，当前白名单：" + ("、".join(merged) if merged else "空（对所有群生效）"))
                if ok
                else "❌ 保存失败，请看插件日志"
            )
            return

        if action in ("清空白名单", "clear"):
            ok = await self._persist(whitelist_groups=[])
            yield event.plain_result("✅ 白名单已清空，现在对所有群和私聊生效" if ok else "❌ 保存失败，请看插件日志")
            return

        yield event.plain_result(
            "用法：\n"
            "/注入前缀 状态\n"
            "/注入前缀 开启 | 关闭\n"
            "/注入前缀 设置 masterpiece, best quality\n"
            "/注入前缀 白名单 群号\n"
            "/注入前缀 移除白名单 群号\n"
            "/注入前缀 清空白名单"
        )



