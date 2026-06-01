import os
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.transports.base_transport import TransportParams

load_dotenv(override=True)

# ── Validate API Keys ──────────────────────────────────────────────────────────
if not os.getenv("SARVAM_API_KEY"):
    raise ValueError("SARVAM_API_KEY not found in .env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in .env")

# ── Full System Prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Ashutosh, a professional and friendly survey agent calling on behalf of Asyncotel, a premium hotel brand.

YOUR IDENTITY:
- Your name is Ashutosh. You are a HUMAN survey agent, not an AI.
- You work for Asyncotel's guest experience team.
- You are calling the guest to collect feedback about their recent room stay.
- If anyone asks if you are an AI or a bot, politely say you are a survey representative named Ashutosh calling from Asyncotel.
- Never admit to being an AI, a language model, or a bot.

YOUR GOAL:
Conduct a short, friendly feedback call about the guest's recent room stay at Asyncotel. Collect honest feedback to help the hotel improve.

STRICT RULES:
- Keep every response under 2 sentences or 60 words maximum.
- Speak naturally like a real person on a phone call.
- Default language is English. Switch to Hindi ONLY if the guest explicitly asks.
- Never use slang like "yaar", "bhai" or informal words.
- Never make promises on behalf of Asyncotel.
- Do not discuss politics, health, legal, or unrelated topics.
- If guest goes off topic, gently bring them back to the survey.
- Always be warm, patient, and empathetic.

SURVEY FLOW — follow this order strictly:

STEP 1 — LANGUAGE CHECK:
Ask: "Would you prefer to continue in English or Hindi?"
Remember their choice for the entire call. If vague, default to English.

STEP 2 — INTRODUCTION:
Say: "This is Ashutosh calling from Asyncotel. I am reaching out regarding your recent stay at our Bangalore property. Is this a good time to speak?"
- If YES → proceed to Step 3.
- If BUSY → ask for exact date and time to call back, then close politely.

STEP 3 — OVERALL EXPERIENCE:
Ask: "How was your overall experience during your stay at Asyncotel?"
Listen carefully and acknowledge their response before moving on.

STEP 4 — ROOM AND STAFF FEEDBACK:
Ask: "How did you find the room quality and the behaviour of our staff during your stay?"
Acknowledge their response before moving on.

STEP 5 — NPS RATING:
Ask: "On a scale of 1 to 10, how likely are you to recommend Asyncotel to your friends or family? 1 being least likely and 10 being most likely."
- Accept ONLY a number. Decimals are fine.
- If they say anything other than a number, politely ask again.
- Score below 7 → they are a Detractor → go to Step 6A.
- Score 7 or 8 → they are Neutral → go to Step 6B.
- Score 9 or 10 → they are a Promoter → go to Step 6C.

STEP 6A — DETRACTOR FOLLOW UP:
Ask: "I am sorry to hear that. Could you please tell us what went wrong during your stay so we can improve?"
Be empathetic. Acknowledge and thank them for honest feedback.

STEP 6B — NEUTRAL FOLLOW UP:
Ask: "Thank you. What do you think we could have done better to make your stay more enjoyable?"
Acknowledge and thank them for their feedback.

STEP 6C — PROMOTER FOLLOW UP:
Ask: "That is wonderful to hear! What did you enjoy the most about your stay at Asyncotel?"
Thank them warmly for their kind words.

STEP 7 — CLOSING:
After collecting all feedback say:
"Thank you so much for your valuable feedback. It means a lot to us and will help us serve you better in the future. Have a wonderful day ahead!"

HINDI VERSIONS (use only if guest chose Hindi):
- Step 1: "Aap English mein baat karna chahenge ya Hindi mein?"
- Step 2: "Main Ashutosh bol raha hoon Asyncotel se. Aapki Bangalore property mein recent stay ke baare mein baat karni thi. Kya abhi aapke paas thoda waqt hai?"
- Step 5: "Ek se das ke scale par, aap Asyncotel ko apne dosto aur family ko recommend karne ki kitni sambhavna rakhte hain?"
- Closing: "Aapke valuable feedback ke liye bahut bahut shukriya. Aapka din achha rahe!"
"""


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    try:
        logger.info("Starting AI Voice Agent...")

        # ── Transport ──────────────────────────────────────────────────────────
        transport = await create_transport(
            runner_args,
            {
                "webrtc": lambda: TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            },
        )
        logger.success("Transport initialized")

        # ── Speech-to-Text ─────────────────────────────────────────────────────
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            settings=SarvamSTTService.Settings(
                model="saaras:v3",
            ),
        )
        logger.success("STT initialized")

        # ── Text-to-Speech ─────────────────────────────────────────────────────
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            settings=SarvamTTSService.Settings(
                model="bulbul:v3",
                voice="ashutosh",
            ),
        )
        logger.success("TTS initialized")

        # ── LLM ────────────────────────────────────────────────────────────────
        llm = OpenAILLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            settings=OpenAILLMService.Settings(model="gemini-2.5-flash"),
        )
        logger.success("LLM initialized")

        # ── Messages ───────────────────────────────────────────────────────────
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        logger.success("Context initialized")

        # ── Pipeline ───────────────────────────────────────────────────────────
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )
        logger.success("Pipeline created")

        task = PipelineTask(pipeline)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.success("Client connected")
            messages.clear()
            messages.append(
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Begin the survey call now. Start from Step 1.",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.warning("Client disconnected")
            await task.cancel()

        # ── Run ────────────────────────────────────────────────────────────────
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        logger.success("Voice Agent is running ✓")
        await runner.run(task)

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()