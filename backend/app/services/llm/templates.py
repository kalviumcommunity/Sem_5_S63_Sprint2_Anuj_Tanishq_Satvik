"""Reusable, versioned prompt template management system for ResearchMate.

Keeps prompt design completely decoupled from application logic while providing:
- Template versioning (e.g., 1.0.0)
- Explicit required and optional variable validation
- Safe variable interpolation resilient to JSON braces
- Role-separated PromptBundle generation
- Centralized template registry
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from backend.app.models.conversation import Message
from backend.app.services.llm.exceptions import (
    MissingPromptVariableError,
    PromptTemplateNotFoundError,
)
from backend.app.services.llm.prompts import (
    PromptBuilder,
    PromptBundle,
    RESEARCHMATE_SYSTEM_PROMPT,
)


class PromptTemplate(BaseModel):
    """Reusable, versioned prompt template with strict variable validation."""

    template_id: str = Field(..., description="Unique identifier for the prompt template")
    name: str = Field(..., description="Human-readable title of the template")
    version: str = Field(default="1.0.0", description="Semantic version string")
    description: str = Field(default="", description="Purpose and operational context of this template")
    system_template: str = Field(..., description="System instructions template")
    user_template: str = Field(..., description="User prompt template with {placeholders}")
    required_variables: List[str] = Field(default_factory=list, description="Mandatory variable keys")
    optional_variables: Dict[str, Any] = Field(default_factory=dict, description="Optional keys with default values")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional tracking metadata")

    def get_declared_variables(self) -> Set[str]:
        """Return all expected variable names (required + optional)."""
        return set(self.required_variables) | set(self.optional_variables.keys())

    def validate_variables(self, variables: Dict[str, Any]) -> None:
        """Ensure all required variables are present and not None."""
        missing = [
            var for var in self.required_variables
            if var not in variables or variables[var] is None
        ]
        if missing:
            raise MissingPromptVariableError(
                template_id=self.template_id,
                missing_vars=missing,
                version=self.version,
            )

    @staticmethod
    def _safe_interpolate(template_str: str, variables: Dict[str, Any]) -> str:
        """Safely substitute known {var} tokens without corrupting literal JSON braces."""
        result = template_str
        for key, value in variables.items():
            str_val = str(value)
            # Replace single brace {key} and double brace {{key}}
            result = re.sub(rf"\{{{key}\}}", lambda _: str_val, result)
            result = re.sub(rf"\{{\{{{key}\}}\}}", lambda _: str_val, result)
        return result

    def render(self, variables: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Render both system and user prompts with variable substitution.

        Returns:
            Tuple[str, str]: (rendered_system_prompt, rendered_user_prompt)
        """
        vars_dict = dict(variables or {})
        # Merge optional defaults with supplied variables
        merged_vars = dict(self.optional_variables)
        merged_vars.update(vars_dict)

        # Validate mandatory variables
        self.validate_variables(merged_vars)

        rendered_system = self._safe_interpolate(self.system_template, merged_vars)
        rendered_user = self._safe_interpolate(self.user_template, merged_vars)

        return rendered_system, rendered_user

    def render_system(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """Render only the system prompt."""
        system_text, _ = self.render(variables)
        return system_text

    def render_user(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """Render only the user prompt."""
        _, user_text = self.render(variables)
        return user_text

    def render_bundle(
        self,
        variables: Optional[Dict[str, Any]] = None,
        history: Optional[List[Message]] = None,
        context: Optional[str] = None,
    ) -> PromptBundle:
        """Render the template into a role-separated PromptBundle."""
        rendered_sys, rendered_usr = self.render(variables)

        builder = PromptBuilder(system_prompt=rendered_sys)

        # Attach context if provided explicitly or in variables
        ctx_val = context or (variables.get("context") if variables else None)
        if ctx_val:
            builder.add_context_message(ctx_val)

        if history:
            builder.add_conversation_history(history)

        builder.add_user_message(rendered_usr)
        return builder.build()


# ==============================================================================
# Canonical ResearchMate Prompt Templates
# ==============================================================================

ACADEMIC_QA_TEMPLATE = PromptTemplate(
    template_id="academic_qa",
    name="Academic Question Answering",
    version="1.0.0",
    description="Scholarly answering template for conceptual breakdowns and academic inquiries.",
    system_template=(
        "You are ResearchMate, an academic research assistant dedicated to precision and scholarly rigor.\n"
        "Field of Discipline: {discipline}\n"
        "Target Academic Level: {academic_level}\n\n"
        "Guidelines:\n"
        "- Provide clear, intellectually rigorous explanations.\n"
        "- Define core technical terminology before expanding on complex concepts.\n"
        "- Highlight prevailing academic consensuses and noted debates in the literature.\n"
        "- Structure answers logically with conceptual clarity."
    ),
    user_template=(
        "Academic Inquiry:\n"
        "{question}\n\n"
        "Please provide a structured, academically rigorous explanation addressing this inquiry."
    ),
    required_variables=["question"],
    optional_variables={
        "discipline": "General Academic",
        "academic_level": "Undergraduate",
    },
    metadata={"author": "ResearchMate Team", "category": "core_qa"},
)

QUERY_REWRITING_TEMPLATE = PromptTemplate(
    template_id="query_rewriting",
    name="Conversational Query Rewriter",
    version="1.0.0",
    description="Transforms conversational multi-turn follow-ups into standalone academic search queries.",
    system_template=(
        "You are an academic query reformulation specialist.\n"
        "Your task is to take a dialogue history and a student's follow-up question, resolve all coreferences "
        "(e.g., 'it', 'this method', 'that study', 'they'), and generate a single, standalone academic search query.\n\n"
        "Rules:\n"
        "- Output ONLY the rewritten search query.\n"
        "- Do NOT include preamble, conversational pleasantries, or explanatory notes.\n"
        "- Keep the query concise, precise, and keyword-rich for vector search (max {max_keywords} keywords).\n"
        "- If the follow-up question is already standalone, preserve its original intent."
    ),
    user_template=(
        "Conversation History:\n"
        "{conversation_history}\n\n"
        "Follow-Up Question: {follow_up_question}\n\n"
        "Standalone Academic Search Query:"
    ),
    required_variables=["conversation_history", "follow_up_question"],
    optional_variables={"max_keywords": 8},
    metadata={"author": "ResearchMate Team", "category": "retrieval"},
)

SUMMARIZATION_TEMPLATE = PromptTemplate(
    template_id="summarization",
    name="Academic Document Summarizer",
    version="1.0.0",
    description="Synthesizes academic papers and text passages into structured research summaries.",
    system_template=(
        "You are an academic literature synthesis specialist.\n"
        "Your role is to produce objective, high-density academic summaries of provided literature.\n\n"
        "Instructions:\n"
        "1. Identify the core thesis, research questions, or objectives.\n"
        "2. Detail the methodology or analytical framework employed.\n"
        "3. Highlight empirical findings, data points, and experimental results.\n"
        "4. Note conclusions, implications, and limitations acknowledged by the authors.\n"
        "5. Keep the summary focused on: {focus}.\n"
        "6. Limit key takeaways to at most {max_bullet_points} bullet points."
    ),
    user_template=(
        "Document Title: {title}\n\n"
        "Source Text:\n"
        "{document_text}\n\n"
        "Generate a structured academic summary adhering strictly to the provided text."
    ),
    required_variables=["document_text"],
    optional_variables={
        "title": "Academic Source",
        "focus": "core methodology and key findings",
        "max_bullet_points": 5,
    },
    metadata={"author": "ResearchMate Team", "category": "summarization"},
)

SOURCE_GROUNDED_ANSWERING_TEMPLATE = PromptTemplate(
    template_id="source_grounded_answering",
    name="Source-Grounded Academic RAG",
    version="1.0.0",
    description="Strictly grounded RAG answering template enforcing citation traceability.",
    system_template=RESEARCHMATE_SYSTEM_PROMPT,
    user_template=(
        "Retrieved Academic Reference Context:\n"
        "--- BEGIN REFERENCE CONTEXT ---\n"
        "{context}\n"
        "--- END REFERENCE CONTEXT ---\n\n"
        "Student Research Question: {question}\n\n"
        "Directives:\n"
        "- Formulate your answer solely from the reference context above.\n"
        "- Attribute all factual assertions with citations adhering to the format: {citation_format}.\n"
        "- If the reference context does not contain sufficient information, state so clearly."
    ),
    required_variables=["context", "question"],
    optional_variables={
        "citation_format": "[Doc: <Title>, Page: <Page>]",
    },
    metadata={"author": "ResearchMate Team", "category": "rag_core"},
)

INSUFFICIENT_CONTEXT_REFUSAL_TEMPLATE = PromptTemplate(
    template_id="insufficient_context_refusal",
    name="Insufficient Context & Refusal Guidance",
    version="1.0.0",
    description="Principled refusal and constructive research guidance when library evidence is missing.",
    system_template=(
        "You are ResearchMate, operating under the core product principle: 'No unsupported answer. No hidden source.'\n\n"
        "Operational Directives:\n"
        "1. When academic evidence retrieved from library holdings is insufficient, never speculate or hallucinate.\n"
        "2. State clearly and politely that available library documents do not contain adequate evidence.\n"
        "3. Identify the specific knowledge gap (what data or topics would be required to answer).\n"
        "4. Provide constructive search recommendations and related academic keywords to assist the student."
    ),
    user_template=(
        "Student Question: {question}\n"
        "Evaluated Library Holdings / Sources: {evaluated_topics}\n\n"
        "Formulate a principled refusal response that explains the evidence gap and suggests alternative "
        "search terms: {suggested_terms}."
    ),
    required_variables=["question"],
    optional_variables={
        "evaluated_topics": "Retrieved library documents and course reserves",
        "suggested_terms": "related academic keywords and domain terms",
    },
    metadata={"author": "ResearchMate Team", "category": "guardrail"},
)


# ==============================================================================
# Centralized Prompt Template Registry
# ==============================================================================

class PromptTemplateRegistry:
    """Central repository for versioned prompt templates."""

    def __init__(self):
        # Maps template_id -> {version_str: PromptTemplate}
        self._registry: Dict[str, Dict[str, PromptTemplate]] = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register canonical ResearchMate templates."""
        for tmpl in [
            ACADEMIC_QA_TEMPLATE,
            QUERY_REWRITING_TEMPLATE,
            SUMMARIZATION_TEMPLATE,
            SOURCE_GROUNDED_ANSWERING_TEMPLATE,
            INSUFFICIENT_CONTEXT_REFUSAL_TEMPLATE,
        ]:
            self.register(tmpl)

    def register(self, template: PromptTemplate) -> None:
        """Register a new template or new version of an existing template."""
        if template.template_id not in self._registry:
            self._registry[template.template_id] = {}
        self._registry[template.template_id][template.version] = template

    def get(self, template_id: str, version: Optional[str] = None) -> PromptTemplate:
        """Retrieve a prompt template by ID and optional version.

        If version is omitted, returns the latest registered version.
        """
        versions_map = self._registry.get(template_id)
        if not versions_map:
            raise PromptTemplateNotFoundError(template_id=template_id, version=version)

        if version:
            template = versions_map.get(version)
            if not template:
                raise PromptTemplateNotFoundError(template_id=template_id, version=version)
            return template

        # Return latest registered version (last registered or highest semver)
        return list(versions_map.values())[-1]

    def list_templates(self) -> List[PromptTemplate]:
        """List the latest version of every registered template."""
        return [list(ver_map.values())[-1] for ver_map in self._registry.values()]

    def list_versions(self, template_id: str) -> List[str]:
        """List all available versions for a given template ID."""
        versions_map = self._registry.get(template_id)
        if not versions_map:
            raise PromptTemplateNotFoundError(template_id=template_id)
        return list(versions_map.keys())

    def reset_to_defaults(self) -> None:
        """Reset registry to default canonical templates (useful for test isolation)."""
        self._registry.clear()
        self._register_default_templates()


# Global singleton instance for application use
prompt_registry = PromptTemplateRegistry()
