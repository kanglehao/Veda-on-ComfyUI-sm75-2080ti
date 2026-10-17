"""User-visible status: text on the Veda node, mirrored to the log.

Uses ComfyUI's native per-node progress text (`send_progress_text`), so no
frontend extension is needed. Safe to call from the sampling thread and
when ComfyUI's server is absent (scripts, tests).

Warnings and errors are bilingual, English then Chinese: the backend cannot
know which language each browser shows, so runtime text carries both
(`bilingual`). Static text (node name, description, tooltips) follows the
UI language through `locales/`; plain status lines stay English.

Errors do not fall back to full attention: a render the user believes is
accelerated must be. `VedaError` stops the run and says why. When the
user can fix it (a missing package, an unsupported GPU, two nodes in the
wrong order) the text says how; otherwise `issue_report` attaches what an
issue needs and where to file it.
"""

from __future__ import annotations

import logging
import traceback

_LOG = logging.getLogger('veda')

ISSUES_URL = 'https://github.com/veda-sparse/Veda-on-ComfyUI/issues/new'


class VedaError(RuntimeError):
    """Veda cannot run this render; the message is shown to the user."""


def bilingual(en: str, zh: str) -> str:
    """A warning or error for the user: the English text, then Chinese."""
    return f'{en}\n{zh}'


def issue_report(en: str, zh: str, context: list[str],
                 error: BaseException | None = None) -> VedaError:
    """A `VedaError` for a failure only we can fix: what happened, then
    the environment, the details and the traceback, ready to paste into
    an issue."""
    lines = [en, zh, '',
             f'Please open an issue at {ISSUES_URL} with everything below.',
             f'请到 {ISSUES_URL} 提 issue，并附上下面的全部内容。',
             '', '--- Veda error report ---', *context]
    if error is not None:
        lines += ['', *(line.rstrip('\n') for line in
                        traceback.format_exception(
                            type(error), error, error.__traceback__))]
    lines.append('--- end of report ---')
    return VedaError('\n'.join(lines))


class NodeStatus:
    """Writes status lines onto one node, skipping exact repeats."""

    def __init__(self, node_id: str | None):
        self.node_id = node_id
        self._last = None

    def show(self, text: str, level: int = logging.INFO,
             zh: str | None = None) -> None:
        """Shows `text` on the node and logs it; `zh`, the Chinese version,
        goes to the node only (the log stays ASCII English)."""
        # Status text carries no emoji, so the only non-ASCII left is the
        # ' · ' separator; the log still gets plain ASCII, because Windows
        # consoles and log files may not be UTF-8. The strip stays as the
        # backstop that keeps that promise whatever a caller passes in.
        plain = text.replace(' · ', ' | ').encode('ascii', 'ignore')
        plain = plain.decode().strip()
        _LOG.log(level, 'Veda: %s', plain)
        if zh:
            text = bilingual(text, zh)
        if text == self._last or self.node_id is None:
            return
        self._last = text
        try:
            from server import PromptServer  # pylint: disable=import-outside-toplevel
            PromptServer.instance.send_progress_text(text, self.node_id)
        except Exception:  # no server (scripts, tests): the log is enough
            pass

    def warn(self, text: str, zh: str | None = None) -> None:
        self.show(text, logging.WARNING, zh)
