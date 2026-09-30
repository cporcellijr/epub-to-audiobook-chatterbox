import os

# The Docker setup turns the speech check on (Whisper); tests that need it supply a fake checker.
os.environ.pop("SPEECH_CHECK_MODEL", None)
