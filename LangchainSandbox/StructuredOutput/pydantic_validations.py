import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")


class LlmSchema(BaseModel):
    reviewer: str = Field(description="Name of the reviewer", default="Anonymous")
    summary: str = Field(description="Brief summary of the review")
    pros: Optional[List[str]] = Field(
        description="Jot down all pros mentioned as a list"
    )
    cons: Optional[List[str]] = Field(
        description="Jot down all cons mentioned as a list"
    )
    sentiment: Literal["pos", "neu", "cons"] = Field(
        description='"Sentiment of the review as positive, neutral or negative"'
    )
    rating: Optional[float] = Field(
        description="Rating of the product if explicitly mentioned in the review"
    )


review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons
Ergonomics limited by physics
Overkill hardware with diminishing returns
Camera processing inconsistency
100x zoom is more marketing than optics
Slow charging relative to competitors

- Goofy
"""

structured_model = model.with_structured_output(LlmSchema)
res = structured_model.invoke(review)

print(json.dumps(dict(res), indent=4, ensure_ascii=False))
