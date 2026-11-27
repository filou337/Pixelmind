"""Answer generation from retrieved sources (dossier §8.3).

ClaudeGenerator passes each source as a `search_result` block with citations enabled:
the API returns, for every claim, which source block it comes from. The model is told to
answer only from those sources and to abstain otherwise.

ExtractiveGenerator needs no LLM: it returns the best passages with their citations
(tests, CI, or no API credentials).
"""

from dataclasses import dataclass, field

PROMPT_VERSION = "rag-v1"
ABSTAIN = "INFORMATION_NON_DISPONIBLE"

SYSTEM_PROMPT = f"""Tu es l'assistant documentaire de PIXELMIND, une plateforme de détection \
de changements de bâtiments entre deux images satellitaires.

Règles :
- Réponds en français, uniquement à partir des résultats de recherche fournis dans le \
message. N'utilise pas tes connaissances générales pour compléter.
- Chaque affirmation doit s'appuyer sur un résultat de recherche cité.
- Si les résultats ne contiennent pas l'information demandée, commence ta réponse par \
{ABSTAIN} puis explique en une phrase ce qui manque. Ne devine pas.
- Les chiffres d'une analyse (surfaces, ratios, nombre de régions, métriques, dates, \
localisations) ne peuvent venir que du résultat dont la source commence par « analysis: ». \
N'en invente jamais et n'en extrapole pas.
- Quand une analyse est en jeu, sépare clairement les faits mesurés, ton interprétation \
et les hypothèses, et rappelle les limites pertinentes.
- Sois concis : quelques phrases ou une courte liste."""


class GenerationError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Source:
    """One search result given to the generator. blocks = citable text units."""

    source: str  # e.g. "docs/adr/0001-….md#Décision" or "analysis:pm_…"
    title: str
    blocks: list[str]
    ref: dict = field(default_factory=dict)  # chunk_id, doc_id, section, page…


@dataclass
class Answer:
    text: str
    abstained: bool
    citations: list[dict]
    provider: str
    model: str | None = None
    refused: bool = False
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)


def _citation(n: int, src: Source, cited_text: str) -> dict:
    return {"n": n, "source": src.source, "title": src.title, "cited_text": cited_text, **src.ref}


class ExtractiveGenerator:
    provider = "extractive"

    def __init__(self, max_passages: int = 3, max_chars: int = 400) -> None:
        self.max_passages, self.max_chars = max_passages, max_chars

    def answer(self, question: str, sources: list[Source]) -> Answer:
        lines, citations = ["Passages les plus pertinents (mode extractif, sans LLM) :"], []
        for n, src in enumerate(sources[: self.max_passages], start=1):
            text = " ".join(src.blocks)
            short = text if len(text) <= self.max_chars else text[: self.max_chars] + "…"
            lines.append(f"- {short} [{n}]")
            citations.append(_citation(n, src, text))
        return Answer("\n".join(lines), abstained=not sources, citations=citations,
                      provider=self.provider)  # fmt: skip


class ClaudeGenerator:
    provider = "claude"

    def __init__(
        self, model: str = "claude-opus-5-5", effort: str = "medium", fallbacks: bool = True,
        max_tokens: int = 8000, client=None,
    ) -> None:  # fmt: skip
        import anthropic

        # credentials: ANTHROPIC_API_KEY, or an `ant auth login` profile
        self.client = client or anthropic.Anthropic()
        self.model, self.effort, self.fallbacks, self.max_tokens = (
            model,
            effort,
            fallbacks,
            max_tokens,
        )

    def _create(self, content: list[dict]):
        import anthropic

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": content}],
        }
        try:
            if self.fallbacks:
                # on a safety decline, the API re-runs the request on a fallback model
                return self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            return self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            raise GenerationError(
                503, "LLM non configuré : clé API Claude absente ou invalide"
            ) from None
        except anthropic.PermissionDeniedError as e:
            raise GenerationError(503, f"accès au modèle refusé : {e.message}") from None
        except anthropic.RateLimitError:
            raise GenerationError(
                429, "limite de débit de l'API Claude atteinte, réessayer"
            ) from None
        except anthropic.BadRequestError as e:
            raise GenerationError(502, f"requête refusée par l'API Claude : {e.message}") from None
        except anthropic.APIStatusError as e:
            raise GenerationError(502, f"erreur API Claude ({e.status_code})") from None
        except anthropic.APIConnectionError:
            raise GenerationError(503, "API Claude injoignable") from None

    def answer(self, question: str, sources: list[Source]) -> Answer:
        content = [
            {
                "type": "search_result",
                "source": s.source,
                "title": s.title,
                "content": [{"type": "text", "text": b} for b in s.blocks],
                "citations": {"enabled": True},
            }
            for s in sources
        ]
        content.append({"type": "text", "text": question})
        resp = self._create(content)

        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        if resp.stop_reason == "refusal":
            return Answer("", abstained=True, citations=[], provider=self.provider,
                          model=resp.model, refused=True, stop_reason="refusal",
                          usage=usage)  # fmt: skip

        parts, cited = [], {}
        for block in resp.content:
            if block.type != "text":
                continue
            marks = []
            for c in block.citations or []:
                if c.type != "search_result_location" or c.search_result_index >= len(sources):
                    continue
                n = c.search_result_index + 1
                cited.setdefault(n, []).append(c.cited_text)
                if f"[{n}]" not in marks:
                    marks.append(f"[{n}]")
            parts.append(block.text + ("".join(f" {m}" for m in marks) if marks else ""))
        text = "".join(parts).strip()
        citations = [
            _citation(n, sources[n - 1], " … ".join(dict.fromkeys(texts)))
            for n, texts in sorted(cited.items())
        ]
        return Answer(
            text, abstained=text.startswith(ABSTAIN), citations=citations, provider=self.provider,
            model=resp.model, stop_reason=resp.stop_reason, usage=usage,
        )  # fmt: skip
