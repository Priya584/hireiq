"""
ConversationHandler — free-form chat over the co-pilot's analysis.

Answers the candidate's questions ("Why did I get only 65%?", "What should I
improve first?", "Is this a good fit for my long-term goals?") using the session
memory_context produced by the workflow. Wired into the UI in Prompt 7.

Usage:
    from memory.chat_handler import ConversationHandler
    handler = ConversationHandler()
    answer = handler.answer_question("Why did I get this score?", memory_context)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the project root importable whether run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.llms import ChatMessage, MessageRole  # noqa: E402
from llama_index.llms.openrouter import OpenRouter  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

# Same free model as the rest of the project (see project memory).
_MODEL = "openai/gpt-oss-120b:free"

_SYSTEM_PROMPT = """\
You are an AI Hiring Co-pilot talking directly to the candidate about their fit
analysis for a specific role. You have a SESSION MEMORY containing everything
known so far: the candidate's resume summary, the job description, the
investigation plan, findings from the job database (SQL) and culture documents
(RAG), the synthesized fit analysis, and any follow-up Q&A.

Rules:
- Answer using ONLY the information in the session memory and the conversation
  so far. Do NOT invent scores, companies, salaries, or facts not present.
- If the memory does not contain something the user asks about (e.g. a numeric
  score that hasn't been computed yet), say so honestly and answer with what the
  analysis DOES show.
- Be direct, specific, and constructive — like a senior recruiter coaching the
  candidate. Reference concrete details from the memory (skills, gaps, market
  data, culture expectations).
- Keep answers focused and practical (a few short paragraphs or bullets).
"""


class ConversationHandler:
    """Holds conversation history and answers questions using memory context."""

    def __init__(self):
        self.history: list[dict] = []

    def add_to_history(self, role: str, content: str) -> None:
        """Append a {"role", "content"} turn to the conversation history."""
        self.history.append({"role": role, "content": content})

    def get_history(self) -> list:
        """Return the full conversation history."""
        return self.history

    def _get_llm(self) -> OpenRouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. Add it to your .env file."
            )
        return OpenRouter(
            api_key=api_key,
            model=_MODEL,
            max_tokens=700,
            temperature=0.3,
        )

    def answer_question(self, question: str, memory_context: str) -> str:
        """
        Answer a free-form question about the analysis using the memory context.

        Records both the question and the answer in the conversation history so
        follow-up questions stay coherent within the session.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"{_SYSTEM_PROMPT}\n\n{memory_context}",
            )
        ]
        # Include prior conversation for continuity.
        for turn in self.history:
            role = (MessageRole.USER if turn["role"] == "user"
                    else MessageRole.ASSISTANT)
            messages.append(ChatMessage(role=role, content=turn["content"]))
        messages.append(ChatMessage(role=MessageRole.USER, content=question))

        try:
            response = self._get_llm().chat(messages)
            answer = str(response.message.content).strip()
        except Exception as exc:
            answer = f"(Sorry — I couldn't answer that right now: {exc})"

        self.add_to_history("user", question)
        self.add_to_history("assistant", answer)
        return answer
