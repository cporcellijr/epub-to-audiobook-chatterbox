class GeneralConfig:
    def __init__(self, args):
        # General arguments
        self.input_file = getattr(args, 'input_file', None)
        self.output_folder = getattr(args, 'output_folder', None)
        self.preview = getattr(args, 'preview', None)
        self.output_text = getattr(args, 'output_text', None)
        self.log = getattr(args, 'log', None)
        self.log_file = None
        self.no_prompt = getattr(args, 'no_prompt', None)
        self.worker_count = getattr(args, 'worker_count', None)
        self.skip_existing = getattr(args, 'skip_existing', None)
        self.use_pydub_merge = getattr(args, 'use_pydub_merge', None)

        # Book parser specific arguments
        self.title_mode = getattr(args, 'title_mode', None)
        self.newline_mode = getattr(args, 'newline_mode', None)
        self.chapter_start = getattr(args, 'chapter_start', None)
        self.chapter_end = getattr(args, 'chapter_end', None)
        # Optional list of 1-based chapter numbers to narrate (within start/end); output is renumbered 1..n
        self.chapter_selection = getattr(args, 'chapter_selection', None)
        self.remove_endnotes = getattr(args, 'remove_endnotes', None)
        self.remove_reference_numbers = getattr(args, 'remove_reference_numbers', None)
        self.search_and_replace_file = getattr(args, 'search_and_replace_file', None)

        # TTS provider: common arguments
        self.tts = getattr(args, 'tts', None)
        self.language = getattr(args, 'language', None)
        self.voice_name = getattr(args, 'voice_name', None)
        self.output_format = getattr(args, 'output_format', None)
        self.model_name = getattr(args, 'model_name', None)

        # OpenAI specific arguments
        self.instructions = getattr(args, 'instructions', None)
        self.speed = getattr(args, 'speed', None)
        # Per-config OpenAI-compatible endpoint (e.g. Kokoro); None = read OPENAI_BASE_URL as before
        self.openai_base_url = getattr(args, 'openai_base_url', None)
        # Paced narration: speak sentence-sized units and insert these pauses (ms); None = off
        self.sentence_pause_ms = getattr(args, 'sentence_pause_ms', None)
        self.paragraph_pause_ms = getattr(args, 'paragraph_pause_ms', None)
        # Paced narration unit granularity: "sentence" (default) or "paragraph" (F-05: one
        # request per paragraph, stretching the model's own inter-sentence gaps instead of one
        # request per sentence)
        self.paced_unit_mode = getattr(args, 'paced_unit_mode', None)
        # Merge finished chapters into one <title>.m4b (chapter markers, cover) instead of loose files
        self.output_m4b = getattr(args, 'output_m4b', None)
        # Multi-voice narration: "single" (default, one voice), "dialogue" (voice_name narrates,
        # dialogue_voice speaks every quoted line) or "cast" (cast_file, a saved cast analysis,
        # gives each attributed speaker a voice; unknown speakers get dialogue_voice)
        self.voice_mode = getattr(args, 'voice_mode', None)
        self.dialogue_voice = getattr(args, 'dialogue_voice', None)
        self.cast_file = getattr(args, 'cast_file', None)
        # Adaptive delivery (Chatterbox only): whispered/shouted-style dialogue (by rule, or the
        # cast's saved moods in cast mode) is read softer/more excited around this book's baseline
        # sliders. The three delivery_* sliders are None = use Chatterbox's saved generation
        # defaults (today's behaviour); when any is set, every request sends the resolved baseline
        # via extra_body even if adaptive_delivery itself is off.
        self.adaptive_delivery = getattr(args, 'adaptive_delivery', None)
        self.delivery_exaggeration = getattr(args, 'delivery_exaggeration', None)
        self.delivery_cfg_weight = getattr(args, 'delivery_cfg_weight', None)
        self.delivery_temperature = getattr(args, 'delivery_temperature', None)
        # Tone matching (Chatterbox only): each voice is turned down wherever it comes out brighter
        # than its own reference clip (core/tone_match.py). None = on; only False turns it off.
        self.tone_match = getattr(args, 'tone_match', None)

        # TTS provider: Azure & Edge TTS specific arguments
        self.break_duration = getattr(args, 'break_duration', None)

        # TTS provider: Edge specific arguments
        self.voice_rate = getattr(args, 'voice_rate', None)
        self.voice_volume = getattr(args, 'voice_volume', None)
        self.voice_pitch = getattr(args, 'voice_pitch', None)
        self.proxy = getattr(args, 'proxy', None)

        # TTS provider: Piper specific arguments
        self.piper_path = getattr(args, 'piper_path', None)
        self.piper_docker_image = getattr(args, 'piper_docker_image', None)
        self.piper_speaker = getattr(args, 'piper_speaker', None)
        self.piper_noise_scale = getattr(args, 'piper_noise_scale', None)
        self.piper_noise_w_scale = getattr(args, 'piper_noise_w_scale', None)
        self.piper_length_scale = getattr(args, 'piper_length_scale', None)
        self.piper_sentence_silence = getattr(args, 'piper_sentence_silence', None)

    def __str__(self):
        return ",\n".join(f"{key}={value}" for key, value in self.__dict__.items())
