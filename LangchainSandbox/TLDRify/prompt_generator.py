from langchain_core.prompts import ChatPromptTemplate


class PromptGenerator:
    """Constructs prompt templates for formatting Chat LLM inputs."""

    def __init__(self) -> None:
        """Initialize the system prompt constraints and templates."""
        try:
            self.prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        Developer: # Objective
                        Answer the user's question using only the supplied context.

                        # Instructions
                            - Use only the information contained in the provided context.
                            - Do not rely on prior knowledge or outside information.
                            - Use chat history only to resolve references such as pronouns or ellipsis.
                            - If the answer is not present in the context, reply exactly:
                                "The provided video summary does not contain information to answer this question."
                            - Do not add any prefixes or suffixes such as "based on context provided".

                        # Output Format
                            - Return exactly one plain-text response and no additional sections or fields.
                            - If the answer is present in the context, output only the answer text.
                            - Otherwise, output exactly the required fallback sentence above.

                        # Context
                            <context>
                            {context}
                            </context>

                            <chat_history>
                            {chat_history}
                            </chat_history>
                """,
                    ),
                    ("human", "{user_message}"),
                ]
            )

        except Exception as e:
            print(f"Error generating prompt template bindings: {e}")
