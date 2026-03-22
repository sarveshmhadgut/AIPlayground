import re
from dotenv import load_dotenv
from youtube_transcript_api import (
    FetchedTranscript,
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)


class YouTubeTranscriptLoader:
    """Fetches and cleans transcripts from YouTube videos."""

    def __init__(self) -> None:
        """Initialize the loader and the YouTube Transcript API."""

        load_dotenv()
        self.api: YouTubeTranscriptApi = YouTubeTranscriptApi()

    def _clean_text(self, text: str) -> str:
        """
        Clean transcript text by removing newlines and bracketed metadata.

        Args:
            text (str): The raw transcript text chunk.

        Returns:
            str: The cleaned text.
        """
        text = text.replace("\n", " ")
        text = re.sub(r"\[.*?\]", "", text)

        return text.strip()

    def fetch(self, video_id: str) -> str:
        """
        Fetch the complete English transcript for a given YouTube video.

        Args:
            video_id (str): The YouTube video ID.

        Returns:
            str: The concatenated transcript text, or an empty string if unavailable.
        """
        try:
            transcripts: FetchedTranscript = self.api.fetch(
                video_id=video_id, languages=["en"]
            )

            text: str = " ".join(
                [
                    self._clean_text(transcript.text)
                    for transcript in transcripts
                    if "♪" not in transcript.text
                    and len(self._clean_text(transcript.text)) > 0
                ]
            )

            return text.strip()

        except TranscriptsDisabled:
            print("No transcript available.")
            return ""

        except Exception as e:
            print(f"Error retrieving transcript: {e}")
            return ""
