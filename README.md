# EPUB to Audiobook — Chatterbox edition

A fork of [p0n1/epub_to_audiobook](https://github.com/p0n1/epub_to_audiobook) for turning EPUBs into
audiobooks with a self-hosted [Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server)
(through its OpenAI-compatible endpoint). It adds a web UI built around that workflow, natural pacing,
single-file M4B output and a book queue. The upstream README follows below; the command-line tool
still works as documented there.

> **Local network only.** Neither the web UI nor Chatterbox has any login. Anyone who can reach port
> 7860 can queue books, write into your audiobook folder and add or delete voices; anyone who can reach
> port 8004 can use and reconfigure Chatterbox. Run it on a home network or behind a VPN, never with
> those ports open to the internet.

This is a personal setup shared as-is: it is tuned for one machine (an RTX 4070 on Docker Desktop for
Windows) and issues or pull requests may not get a response.

The repo holds the whole stack:

| Path | What |
|---|---|
| repo root | The audiobook app (this fork of epub_to_audiobook) |
| [`chatterbox/`](chatterbox/) | Chatterbox-TTS-Server, added as a git subtree at upstream commit 915ae28, plus one commit of local patches (AAC/FLAC output, cleaner speed changes, faster generation, a GPU memory-leak fix) |
| [`docker-compose.chatterbox.yml`](docker-compose.chatterbox.yml) | Builds and runs both as two containers |
| [`docs/chatterbox-edition/`](docs/chatterbox-edition/) | Work log, findings and the code-review brief |

## What this fork adds

**Web UI** (`audiobook_generator/ui/chatterbox_ui.py`, served by `main_ui.py`)
- **Book picker**: searchable dropdown of the EPUBs in a mounted library, labelled "Title — Author"
  from each book's metadata (cached; new books are picked up automatically). Uploading an EPUB still works.
- **Chapter checkboxes** with automatic selection: story chapters are ticked; title pages, copyright,
  contents, dedications, acknowledgements, "about the author", newsletter pages and similar are unticked.
  Shows each chapter's opening words and estimated listening time. Ticked chapters are numbered 1…n.
- **Voice dropdown** filled from Chatterbox's voices folder. An optional **Engine** choice next to it
  switches between Chatterbox and [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (only
  shown when `KOKORO_BASE_URL` is configured); the Voice dropdown swaps to Kokoro's English voices,
  best-graded first. A **Sample** button next to the Voice dropdown speaks a short fixed phrase with
  the selected engine, voice and speed so a voice can be auditioned before queuing a book.
- **Voice lab**: play a phrase with any voice, tune Chatterbox's exaggeration / CFG weight / temperature
  and save them as the settings books use, add new voices (long pauses in the sample are removed,
  because Chatterbox copies a reference clip's pauses), and delete a voice you added (Chatterbox's
  built-in voices can't be deleted; a browser confirmation is required, and a voice a queued or
  running book still uses is refused).
- **Queue**: add books with their own voice, chapters and pauses; they run one at a time in order.
  In Cast mode, choose chapters first, then **Analyse selected chapters**. This holds audiobook jobs
  while cast analyses run one at a time, so you can analyse a cast, queue its book, and repeat before
  pressing **Start queued books**. A book already generating
  finishes before cast analysis starts; audio generation and cast analysis never overlap. Less-used
  pause, resume, stop, remove and retry controls are under **More queue actions**. The queue survives
  restarts and resumes an interrupted book.

**Narration and output**
- **Paced narration**: paragraphs are taken from the EPUB's HTML, text is sent a sentence at a time,
  and real pauses are inserted after sentences and between paragraphs (adjustable).
- **Multi-voice narration** (Voice mode on the Make tab). *Single voice* reads everything with one
  voice, exactly as before. *Narrator + dialogue voice* gives every quoted line a second voice; no LLM
  needed. *Cast* asks a local LLM (any OpenAI-compatible chat endpoint: Ollama, llama.cpp, LM Studio;
  book text never leaves the machine) who speaks each line: **Analyse selected chapters** queues the pass as a job
  of its own, Chatterbox's model is unloaded from the GPU while it runs and reloaded after, and the
  result is an editable cast table (character, lines, gender, voice, sample button). To help pick
  voices, the same analysis writes a short profile of each of the most-spoken characters from the
  book's own passages (role, who they are, relationships, a "sounds like" casting note, and the kind
  of voice that fits: pitch, huskiness, liveliness), shown in the table and under it when you click a
  character, with the first line they speak. Voices are suggested by gender (Kokoro ids carry it;
  Chatterbox voices get theirs from a **Voice gender** setting in the Voice lab) and, for Chatterbox,
  by how each voice measures against the profile: **Measure voices** in the Voice lab has each voice
  speak one sentence and measures its pitch, huskiness and liveliness (a voice you add is measured
  straight away). The main characters stay distinct from each other and the narrator; lines whose
  speaker the LLM couldn't tell get the dialogue voice. **Auto-pick suggested voices** (next to
  Analyse) makes a re-analysis keep only the voices you saved yourself; **Suggest voices again**
  re-matches an existing cast. The cast is saved per book under
  `casts/` in the app data folder, and **Add to queue** carries a snapshot of it with the job. Units
  never span a change of voice; the pause between narration and a quote is the sentence pause.
  Validation on a real LLM: `docs/chatterbox-edition/experiments/multivoice/`.
- **Adaptive delivery** (Chatterbox only, Make tab checkbox, on by default): dialogue tagged
  whispered/murmured/shouted/screamed and the like (rule-based, or the cast's own read in Cast mode)
  is spoken softer and quieter, or more excited and a little louder, around this book's baseline
  sliders (its Voice lab exaggeration/CFG/temperature, shown live under the checkbox) instead of one
  flat delivery for the whole book. A per-book baseline (the Voice lab sliders at the moment you
  **Add to queue**) is sent even with adaptive delivery switched off, so a book can have its own
  "voice" without the mood swings. Every adaptive line gets a peak guard so an excited take can never
  clip. Voice lab has a **▶ Play soft / normal / excited** button to audition the three deliveries
  around the current sliders before queuing a book. See `docs/chatterbox-edition/WORKLOG.md` #14 for
  the presets, the rule cues and a measured LLM-mood accuracy trade-off.
- **Single M4B** per book with chapter markers, cover art, title/author tags. Chapters are generated in a
  hidden `.chapters` folder and merged only when all succeed, so a library scanner never sees a
  half-finished book; a failed book can be resumed with "Skip chapters already made".

**Fixes to the upstream pipeline**
- Chapters follow the EPUB's reading order (spine) instead of its file list, so the table of contents
  is no longer narrated and chapters can't come out of order.
- Chapter files are written under a temporary name and renamed when complete.
- The output folder is filled in from the book's title; previewing writes nothing.

**Merged upstream pull requests**: #191 skip existing chapters and resource-leak fixes (@gumpgit),
#189 larger chapter ranges (@bpothier), #186 cover, title and author from the EPUB (@hihilla).

## Running it

Needs Docker with an NVIDIA GPU (Chatterbox's Original model uses about 4.5 GB of VRAM).

1. Copy [`.env.example`](.env.example) to `.env` and set the paths. For a first run, copy
   `chatterbox/config.audiobook.yaml` **as** `config.yaml`, plus `chatterbox/voices/`, into the
   folder you set as `CHATTERBOX_DATA`. (`chatterbox/config.yaml` itself is the untouched
   upstream template — Turbo model, seed 0 — not what this fork is tuned for.) If Chatterbox's
   logs say its config path is a directory, the copy didn't happen before `docker compose up`:
   Docker created an empty directory for the missing bind-mount file; copy the file into place,
   remove that empty directory, and restart.
2. Create the Docker network once (`docker network create tts`), or point `DOCKER_NETWORK` at an existing one.
3. Build and start both containers:
   ```
   docker compose -f docker-compose.chatterbox.yml up -d --build
   ```
   The first build of the Chatterbox image downloads CUDA and PyTorch and takes a while; later
   builds only redo the layers that changed.
4. Open http://localhost:7860 (Chatterbox's own UI and API are on port 8004).

Faster generation (opt-in): `TTS_COMPILE=on` in `.env` makes Chatterbox run the Original model's
per-token step through `torch.compile` with CUDA graphs (`chatterbox/fast_t3.py`), measured about
2.3x faster whole requests with the same model output. It compiles while the model loads (about 17 s
more start-up) and falls back to the stock loop whenever it doesn't apply. It ships off; run
`docs/chatterbox-edition/experiments/f45/validate_build.py` on your GPU first (see that folder's
README) and turn it on only when every check passes.

Updating Chatterbox from upstream: `git subtree pull --prefix=chatterbox
https://github.com/devnen/Chatterbox-TTS-Server.git main --squash`, then check the local patches still
apply. `chatterbox/patches/apply_speed_patches.py` stops the image build if its target code changed.

Chatterbox's own web page (port 8004) has its own Save button for delivery settings
(temperature, exaggeration, CFG weight, seed, speed, language): it posts back the six values
the page loaded when it was opened, so leaving that tab open and clicking its Save can silently
revert whatever this app's Voice lab saved afterwards. Use the app's Voice lab for delivery
settings; if you do use the Chatterbox page directly, reload it first. Separately,
`chatterbox/start.py` / `start.sh` / `start.bat` are upstream's own launchers and aren't used by
this build — the Docker image's `CMD` runs `server.py` directly.

Settings the app reads (the compose file sets them):

| Environment variable | Purpose |
|---|---|
| `OPENAI_BASE_URL` | Chatterbox's OpenAI endpoint, e.g. `http://chatterbox:8004/v1` |
| `OPENAI_API_KEY` | Any non-empty value (Chatterbox has no auth) |
| `TTS_VOICES_DIR` | Chatterbox's voices folder (writable, for the Voice lab) |
| `OPENAI_DEFAULT_VOICE` | Voice selected by default |
| `CHATTERBOX_CONFIG` | Chatterbox's `config.yaml` (read-only), for the Voice lab sliders |
| `CHATTERBOX_URL` | Chatterbox root URL, if it isn't `OPENAI_BASE_URL` minus `/v1` |
| `EBOOK_LIBRARY_DIR` | Ebook library for the book picker |
| `KOKORO_BASE_URL` | Optional second engine's OpenAI endpoint, e.g. `http://kokoro:8880/v1`. Leave unset to hide the Engine choice entirely |
| `KOKORO_DEFAULT_VOICE` | Kokoro voice id selected by default (default: the server's own `default_voice`, else `af_heart`) |
| `LLM_BASE_URL` | Optional local OpenAI-compatible chat endpoint for multi-voice cast analysis, e.g. `http://ollama:11434/v1`. Leave unset to hide the Cast voice mode |
| `LLM_MODEL` | Chat model name for cast analysis |
| `LLM_API_KEY` | Key for that endpoint, if it wants one |
| `LLM_UNLOAD_CHATTERBOX` | `on` (default) frees Chatterbox's GPU memory during a cast analysis and reloads it after; `off` leaves it loaded |

Books are written to `audiobook_output/<title>/` inside the container; mount your audiobook library there.

Tests: `python -m unittest discover -s tests -t . -p "*test*.py"` (ffmpeg required).

## Credits

MIT licensed, like upstream. `chatterbox/` is Chatterbox-TTS-Server by devnen (MIT, see
`chatterbox/LICENSE`). The chapter auto-selection scoring is adapted from
[abogen](https://github.com/denizsafak/abogen) (MIT); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

# EPUB to Audiobook Converter [![Discord](https://img.shields.io/discord/1177631634724491385?label=Discord&logo=discord&logoColor=white)](https://discord.com/invite/pgp2G8zhS7) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/p0n1/epub_to_audiobook)

*Join our [Discord](https://discord.com/invite/pgp2G8zhS7) server for any questions or discussions. You can also ask questions about this project on [DeepWiki](https://deepwiki.com/p0n1/epub_to_audiobook).*

This project provides a command-line tool to convert EPUB ebooks into audiobooks. It now supports both the [Microsoft Azure Text-to-Speech API](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/rest-text-to-speech) (alternativly [EdgeTTS](https://github.com/rany2/edge-tts)) and the [OpenAI Text-to-Speech API](https://platform.openai.com/docs/guides/text-to-speech) to generate the audio for each chapter in the ebook. The output audio files are optimized for use with [Audiobookshelf](https://github.com/advplyr/audiobookshelf).

<!-- *This project was developed with the help of ChatGPT.* -->

## Recent Updates

- 2025-05-23: Added a web interface (WebUI) to the project.

## Audio Sample

If you're interested in hearing a sample of the audiobook generated by this tool, check the links bellow. 

- [Azure TTS Sample](https://audio.com/paudi/audio/0008-chapter-vii-agricultural-experience)
- [OpenAI TTS Sample](https://audio.com/paudi/audio/openai-0008-chapter-vii-agricultural-experience-i-had-now-been-in)
- Edge TTS Sample: the voice is almost the same as Azure TTS
- [Piper TTS](https://rhasspy.github.io/piper-samples/)
- [Kokoro TTS](https://huggingface.co/spaces/hexgrad/Kokoro-TTS) (usage of this is done through a local OpenAI endpoint)

## Requirements

- Python 3.10+ Or ***Docker***
- For using *Azure TTS*, A Microsoft Azure account with access to the [Microsoft Cognitive Services Speech Services](https://portal.azure.com/#create/Microsoft.CognitiveServicesSpeechServices) is required.
- For using *OpenAI TTS*, OpenAI [API Key](https://platform.openai.com/api-keys) is required.
  - If you are using Kokoro TTS, you won't need an official OpenAI key, but you will need to put a dummy value in the env for it. (e.g. `export OPENAI_API_KEY='fake'`) unless you are using the docker compose file (see below)
- For using *Edge TTS*, no API Key is required.
- Piper TTS executable and models for *Piper TTS*

## Audiobookshelf Integration

The audiobooks generated by this project are optimized for use with [Audiobookshelf](https://github.com/advplyr/audiobookshelf). Each chapter in the EPUB file is converted into a separate MP3 file, with the chapter title extracted and included as metadata.

![demo](./examples/audiobookshelf.png)

### Chapter Titles

Parsing and extracting chapter titles from EPUB files can be challenging, as the format and structure may vary significantly between different ebooks. The script employs a simple but effective method for extracting chapter titles, which works for most EPUB files. The method involves parsing the EPUB file and looking for the `title` tag in the HTML content of each chapter. If the title tag is not present, a fallback title is generated using the first few words of the chapter text.

Please note that this approach may not work perfectly for all EPUB files, especially those with complex or unusual formatting. However, in most cases, it provides a reliable way to extract chapter titles for use in Audiobookshelf.

When you import the generated MP3 files into Audiobookshelf, the chapter titles will be displayed, making it easy to navigate between chapters and enhancing your listening experience.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/p0n1/epub_to_audiobook.git
    cd epub_to_audiobook
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Note: Python 3.14 requires the updated dependency set in this repository. Older installs pinned `gradio==5.33.1`, which could force a `pydantic-core` source build and fail with a PyO3 compatibility error during installation.

4. Set the following environment variables with your Azure Text-to-Speech API credentials, or your OpenAI API key if you're using OpenAI TTS:

    ```bash
    export MS_TTS_KEY=<your_subscription_key> # for Azure
    export MS_TTS_REGION=<your_region> # for Azure
    export OPENAI_API_KEY=<your_openai_api_key> # for OpenAI
    ```

## Web Interface (WebUI)

For users who prefer a graphical interface, this project includes a web-based UI built with Gradio. The WebUI provides an intuitive way to configure all the options and convert your EPUB files without using the command line.

![WebUI Screenshot](./examples/webui.png)

### Environment Variables for WebUI

The WebUI respects the same environment variables as the command-line tool:

```bash
export MS_TTS_KEY=<your_subscription_key>      # For Azure TTS
export MS_TTS_REGION=<your_region>             # For Azure TTS
export OPENAI_API_KEY=<your_openai_api_key>    # For OpenAI TTS
export OPENAI_BASE_URL=<custom_endpoint>       # Optional: For custom OpenAI-compatible endpoints
```

Make sure to set the environment variables for the service you are using before starting the WebUI.

### Starting the WebUI

Make sure you have followed the [Installation](#installation) steps before starting the WebUI.

To launch the web interface, run:

```bash
python3 main_ui.py
```

By default, the WebUI will be available at `http://127.0.0.1:7860`. You can customize the host and port:

```bash
python3 main_ui.py --host 127.0.0.1 --port 8080
```

Remember to press `Ctrl+C` in the terminal to stop the server if you want to stop it after you are done.

### WebUI Features

The web interface provides:

- **File Upload**: Drag and drop your EPUB file directly into the browser
- **TTS Provider Selection**: Easy switching between Azure, OpenAI, Edge, and Piper TTS with provider-specific options
- **Voice Configuration**: Dropdown menus for selecting languages, voices, and output formats
- **Advanced Settings**: All command-line options are available through the web interface
- **Real-time Logs**: View conversion progress and logs directly in the browser
- **Preview Mode**: Test your settings without generating audio
- **Search & Replace**: Upload text replacement files for pronunciation fixes

### Using the WebUI

1. **Upload your EPUB file** using the file selector
2. **Choose your TTS provider** from the tabs (OpenAI, Azure, Edge, or Piper)
3. **Configure provider-specific settings**:
   - **OpenAI**: Select model, voice, speed, and format
   - **Azure**: Choose language, voice, format, and break duration
   - **Edge**: Set language, voice, rate, volume, and pitch
   - **Piper**: Configure local or Docker deployment with voice options
4. **Set output directory** or use the default timestamped folder
5. **Adjust advanced options** if needed (chapter range, text processing, etc.)
6. **Click Start** to begin conversion
7. **Monitor progress** through the integrated log viewer

You can select a few chapters to preview the audio before starting the full conversion.

### Docker with WebUI (The Easiest Way If You Are Familiar With Docker)

You can also run the WebUI using Docker. Use the provided `docker-compose.webui.yml` file. Make sure to edit the file with your API keys for your TTS provider.

```bash
# Edit docker-compose.webui.yml with your API keys
docker compose -f docker-compose.webui.yml up
```

The WebUI will be accessible at `http://localhost:7860` or `http://127.0.0.1:7860`.

### Security Considerations of WebUI

The WebUI is a web application that runs on your local machine. It's currently not designed to be accessible from the open internet. There is no authorization mechanism in place. So you should not expose it to the open internet otherwise it would lead to unauthorized access to your TTS providers.

## Usage

To convert an EPUB ebook to an audiobook, run the following command, specifying the TTS provider of your choice with the `--tts` option:

```bash
python3 main.py <input_file> <output_folder> [options]
```

To check the latest option descriptions for this script, you can run the following command in the terminal:

```bash
python3 main.py -h
```

```bash
usage: main.py [-h] [--tts {azure,openai,edge,piper}]
               [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--preview]
               [--no_prompt] [--language LANGUAGE]
               [--newline_mode {single,double,none}]
               [--title_mode {auto,tag_text,first_few}]
               [--chapter_start CHAPTER_START] [--chapter_end CHAPTER_END]
               [--output_text] [--remove_endnotes]
               [--search_and_replace_file SEARCH_AND_REPLACE_FILE]
               [--worker_count WORKER_COUNT] [--skip_existing]
               [--voice_name VOICE_NAME] [--output_format OUTPUT_FORMAT]
               [--model_name MODEL_NAME] [--voice_rate VOICE_RATE]
               [--voice_volume VOICE_VOLUME] [--voice_pitch VOICE_PITCH]
               [--proxy PROXY] [--break_duration BREAK_DURATION]
               [--piper_path PIPER_PATH] [--piper_speaker PIPER_SPEAKER]
               [--piper_sentence_silence PIPER_SENTENCE_SILENCE]
               [--piper_length_scale PIPER_LENGTH_SCALE]
               input_file output_folder

Convert text book to audiobook

positional arguments:
  input_file            Path to the EPUB file
  output_folder         Path to the output folder

options:
  -h, --help            show this help message and exit
  --tts {azure,openai,edge,piper}
                        Choose TTS provider (default: azure). azure: Azure
                        Cognitive Services, openai: OpenAI TTS API. When using
                        azure, environment variables MS_TTS_KEY and
                        MS_TTS_REGION must be set. When using openai,
                        environment variable OPENAI_API_KEY must be set.
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Log level (default: INFO), can be DEBUG, INFO,
                        WARNING, ERROR, CRITICAL
  --preview             Enable preview mode. In preview mode, the script will
                        not convert the text to speech. Instead, it will print
                        the chapter index, titles, and character counts.
  --no_prompt           Don't ask the user if they wish to continue after
                        estimating the cloud cost for TTS. Useful for
                        scripting.
  --language LANGUAGE   Language for the text-to-speech service (default: en-
                        US). For Azure TTS (--tts=azure), check
                        https://learn.microsoft.com/en-us/azure/ai-
                        services/speech-service/language-
                        support?tabs=tts#text-to-speech for supported
                        languages. For OpenAI TTS (--tts=openai), their API
                        detects the language automatically. But setting this
                        will also help on splitting the text into chunks with
                        different strategies in this tool, especially for
                        Chinese characters. For Chinese books, use zh-CN, zh-
                        TW, or zh-HK.
  --newline_mode {single,double,none}
                        Choose the mode of detecting new paragraphs: 'single',
                        'double', or 'none'. 'single' means a single newline
                        character, while 'double' means two consecutive
                        newline characters. 'none' means all newline
                        characters will be replace with blank so paragraphs
                        will not be detected. (default: double, works for most
                        ebooks but will detect less paragraphs for some
                        ebooks)
  --title_mode {auto,tag_text,first_few}
                        Choose the parse mode for chapter title, 'tag_text'
                        search 'title','h1','h2','h3' tag for title,
                        'first_few' set first 60 characters as title, 'auto'
                        auto apply the best mode for current chapter.
  --chapter_start CHAPTER_START
                        Chapter start index (default: 1, starting from 1)
  --chapter_end CHAPTER_END
                        Chapter end index (default: -1, meaning to the last
                        chapter)
  --output_text         Enable Output Text. This will export a plain text file
                        for each chapter specified and write the files to the
                        output folder specified.
  --remove_endnotes     This will remove endnote numbers from the end or
                        middle of sentences. This is useful for academic
                        books.
  --skip_existing       Skip generating audio for a chapter if its output mp3
                        file already exists in the output folder. Useful for
                        resuming interrupted conversions.
  --search_and_replace_file SEARCH_AND_REPLACE_FILE
                        Path to a file that contains 1 regex replace per line,
                        to help with fixing pronunciations, etc. The format
                        is: <search>==<replace> Note that you may have to
                        specify word boundaries, to avoid replacing parts of
                        words.
  --worker_count WORKER_COUNT
                        Specifies the number of parallel workers to use for 
                        audiobook generation. Increasing this value can 
                        significantly speed up the process by processing 
                        multiple chapters simultaneously. Note: Chapters may 
                        not be processed in sequential order, but this will 
                        not affect the final audiobook.

  --voice_name VOICE_NAME
                        Various TTS providers has different voice names, look
                        up for your provider settings.
  --output_format OUTPUT_FORMAT
                        Output format for the text-to-speech service.
                        Supported format depends on selected TTS provider
  --model_name MODEL_NAME
                        Various TTS providers has different neural model names

openai specific:
  --speed SPEED         The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.
  --instructions INSTRUCTIONS
                        Instructions for the TTS model. Only supported for 'gpt-4o-mini-tts' model.

edge specific:
  --voice_rate VOICE_RATE
                        Speaking rate of the text. Valid relative values range
                        from -50%(--xxx='-50%') to +100%. For negative value
                        use format --arg=value,
  --voice_volume VOICE_VOLUME
                        Volume level of the speaking voice. Valid relative
                        values floor to -100%. For negative value use format
                        --arg=value,
  --voice_pitch VOICE_PITCH
                        Baseline pitch for the text.Valid relative values like
                        -80Hz,+50Hz, pitch changes should be within 0.5 to 1.5
                        times the original audio. For negative value use
                        format --arg=value,
  --proxy PROXY         Proxy server for the TTS provider. Format:
                        http://[username:password@]proxy.server:port

azure/edge specific:
  --break_duration BREAK_DURATION
                        Break duration in milliseconds for the different
                        paragraphs or sections (default: 1250, means 1.25 s).
                        Valid values range from 0 to 5000 milliseconds for
                        Azure TTS.

piper specific:
  --piper_path PIPER_PATH
                        Path to the Piper TTS executable
  --piper_speaker PIPER_SPEAKER
                        Piper speaker id, used for multi-speaker models
  --piper_sentence_silence PIPER_SENTENCE_SILENCE
                        Seconds of silence after each sentence
  --piper_length_scale PIPER_LENGTH_SCALE
                        Phoneme length, a.k.a. speaking rate
```  

**Example**:

```bash
python3 main.py examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub output_folder
```

Executing the above command will generate a directory named `output_folder` and save the MP3 files for each chapter inside it using default TTS provider and voice. Once generated, you can import these audio files into [Audiobookshelf](https://github.com/advplyr/audiobookshelf) or play them with any audio player of your choice.

## Preview Mode

Before converting your epub file to an audiobook, you can use the `--preview` option to get a summary of each chapter. This will provide you with the character count of each chapter and the total count, instead of converting the text to speech.

**Example**:

```bash
python3 main.py examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub output_folder --preview
```

## Search & Replace

You may want to search and replace text, either to expand abbreviations, or to help with pronunciation. You can do this by specifying a search and replace file, which contains a single regex search and replace per line, separated by '==':

**Example**:

**search.conf**:

```text
# this is the general structure
<search>==<replace>
# this is a comment
# fix cardinal direction abbreviations
N\.E\.==north east
# be careful with your regexes, as this would also match Sally N. Smith
N\.==north
# pronounce Barbadoes like the locals
Barbadoes==Barbayduss
```

```bash
python3 main.py examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub output_folder --search_and_replace_file search.conf
```

**Example**:

```bash
python3 main.py examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub output_folder --preview
```

## Using with Docker

This tool is available as a Docker image, making it easy to run without needing to manage Python dependencies.

First, make sure you have Docker installed on your system.

You can pull the Docker image from the GitHub Container Registry:

```bash
docker pull ghcr.io/p0n1/epub_to_audiobook:latest
```

Then, you can run the tool with the following command:

```bash
docker run -i -t --rm -v ./:/app -e MS_TTS_KEY=$MS_TTS_KEY -e MS_TTS_REGION=$MS_TTS_REGION ghcr.io/p0n1/epub_to_audiobook your_book.epub audiobook_output --tts azure
```

For OpenAI, you can run:

```bash
docker run -i -t --rm -v ./:/app -e OPENAI_API_KEY=$OPENAI_API_KEY ghcr.io/p0n1/epub_to_audiobook your_book.epub audiobook_output --tts openai
```

Replace `$MS_TTS_KEY` and `$MS_TTS_REGION` with your Azure Text-to-Speech API credentials. Replace `$OPENAI_API_KEY` with your OpenAI API key. Replace `your_book.epub` with the name of the input EPUB file, and `audiobook_output` with the name of the directory where you want to save the output files.

The `-v ./:/app` option mounts the current directory (`.`) to the `/app` directory in the Docker container. This allows the tool to read the input file and write the output files to your local file system.

The `-i` and `-t` options are required to enable interactive mode and allocate a pseudo-TTY.

**You can also check the [this example config file](./docker-compose.example.yml) for docker compose usage.**

## User-Friendly Guide for Windows Users

For Windows users, especially if you're not very familiar with command-line tools, we've got you covered. We understand the challenges and have created a guide specifically tailored for you.

Check this [step by step guide](https://gist.github.com/p0n1/cba98859cdb6331cc1aab835d62e4fba) and leave a message if you encounter issues.

## How to Get Your Azure Cognitive Service Key?

- Azure subscription - [Create one for free](https://azure.microsoft.com/free/cognitive-services)
- [Create a Speech resource](https://portal.azure.com/#create/Microsoft.CognitiveServicesSpeechServices) in the Azure portal.
- Get the Speech resource key and region. After your Speech resource is deployed, select **Go to resource** to view and manage keys. For more information about Cognitive Services resources, see [Get the keys for your resource](https://learn.microsoft.com/en-us/azure/cognitive-services/cognitive-services-apis-create-account#get-the-keys-for-your-resource).

*Source: <https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech#prerequisites>*

## How to Get Your OpenAI API Key?

Check https://platform.openai.com/docs/quickstart/account-setup. Make sure you check the [price](https://openai.com/pricing) details before use.

## ✨ About Edge TTS

Edge TTS and Azure TTS are almost same, the difference is that Edge TTS don't require API Key because it's based on Edge read aloud functionality, and parameters are restricted a bit, like [custom ssml](https://github.com/rany2/edge-tts#custom-ssml).

Check https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462 for supported voices.

**If you want to try this project quickly, Edge TTS is highly recommended.**

## Customization of Voice and Language

You can customize the voice and language used for the Text-to-Speech conversion by passing the `--voice_name` and `--language` options when running the script.

Microsoft Azure offers a range of voices and languages for the Text-to-Speech service. For a list of available options, consult the [Microsoft Azure Text-to-Speech documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support?tabs=tts#text-to-speech).

You can also listen to samples of the available voices in the [Azure TTS Voice Gallery](https://aka.ms/speechstudio/voicegallery) to help you choose the best voice for your audiobook.

For example, if you want to use a British English female voice for the conversion, you can use the following command:

```bash
python3 main.py <input_file> <output_folder> --voice_name en-GB-LibbyNeural --language en-GB
```

For OpenAI TTS, you can specify the model, voice, and format options using `--model_name`, `--voice_name`, and `--output_format`, respectively.

## More examples

Here are some examples that demonstrate various option combinations:

### Examples Using Azure TTS

1. **Basic conversion using Azure with default settings**  
   This command will convert an EPUB file to an audiobook using Azure's default TTS settings.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts azure
   ```

2. **Azure conversion with custom language, voice and logging level**  
   Converts an EPUB file to an audiobook with a specified voice and a custom log level for debugging purposes.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts azure --language zh-CN --voice_name "zh-CN-YunyeNeural" --log DEBUG
   ```

3. **Azure conversion with chapter range and break duration**  
   Converts a specified range of chapters from an EPUB file to an audiobook with custom break duration between paragraphs.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts azure --chapter_start 5 --chapter_end 10 --break_duration "1500"
   ```

### Examples Using OpenAI TTS

1. **Basic conversion using OpenAI with default settings**  
   This command will convert an EPUB file to an audiobook using OpenAI's default TTS settings.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts openai
   ```

2. **OpenAI conversion with HD model and specific voice**  
   Converts an EPUB file to an audiobook using the high-definition OpenAI model and a specific voice choice.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts openai --model_name "tts-1-hd" --voice_name "fable"
   ```

3. **OpenAI conversion with preview and text output**  
   Enables preview mode and text output, which will display the chapter index and titles instead of converting them and will also export the text.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts openai --preview --output_text
   ```

## Example using an OpenAI-compatible service

It is possible to use an OpenAI-compatible service, like [matatonic/openedai-speech](https://github.com/matatonic/openedai-speech). In that case, it **is required** to set the `OPENAI_BASE_URL` environment variable, otherwise it would just default to the standard OpenAI service. While the compatible service might not require an API key, the OpenAI client still does, so make sure to set it to something nonsensical.

If your OpenAI-compatible service is running on `http://127.0.0.1:8000` and you have added a custom voice named `skippy`, you can use the following command:

```shell
docker run -i -t --rm -v ./:/app -e OPENAI_BASE_URL=http://127.0.0.1:8000/v1 -e OPENAI_API_KEY=nope ghcr.io/p0n1/epub_to_audiobook your_book.epub audiobook_output --tts openai --voice_name=skippy --model_name=tts-1-hd
```

Scroll down to the Kokoro TTS example below to see a more specific example of this.

### Examples Using Edge TTS

1. **Basic conversion using Edge with default settings**  
   This command will convert an EPUB file to an audiobook using Edge's default TTS settings.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts edge
   ```

2. **Edge conversion with custom language, voice and logging level**
   Converts an EPUB file to an audiobook with a specified voice and a custom log level for debugging purposes.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts edge --language zh-CN --voice_name "zh-CN-YunxiNeural" --log DEBUG
   ```

3. **Edge conversion with chapter range and break duration**
   Converts a specified range of chapters from an EPUB file to an audiobook with custom break duration between paragraphs.

   ```sh
   python3 main.py "path/to/book.epub" "path/to/output/folder" --tts edge --chapter_start 5 --chapter_end 10 --break_duration "1500"
   ```

### Examples Using Piper TTS

*Make sure you have installed Piper TTS and have an onnx model file and corresponding config file. Check [Piper TTS](https://github.com/rhasspy/piper) for more details. You can follow their instructions to install Piper TTS, download the models and config files, play with it and then come back to try the examples below.*

This command will convert an EPUB file to an audiobook using Piper TTS using the bare minimum parameters.
You always need to specify an onnx model file and the `piper` executable needs to be in the current $PATH. 

```sh
python3 main.py "path/to/book.epub" "path/to/output/folder" --tts piper --model_name <path_to>/en_US-libritts_r-medium.onnx
```

You can specify your custom path to the piper executable by using the `--piper_path` parameter.

```sh
python3 main.py "path/to/book.epub" "path/to/output/folder" --tts piper --model_name <path_to>/en_US-libritts_r-medium.onnx --piper_path <path_to>/piper
```

Some models support multiple voices and that can be specified by using the voice_name parameter.

```sh
python3 main.py "path/to/book.epub" "path/to/output/folder" --tts piper --model_name <path_to>/en_US-libritts_r-medium.onnx --piper_speaker 256
```

You can also specify speed (piper_length_scale) and pause duration (piper_sentence_silence).

```sh
python3 main.py "path/to/book.epub" "path/to/output/folder" --tts piper --model_name <path_to>/en_US-libritts_r-medium.onnx --piper_speaker 256 --piper_length_scale 1.5 --piper_sentence_silence 0.5
```

Piper TTS outputs `wav` format files (or raw) by default you should be able to specify any reasonable format via the `--output_format` parameter. The `opus` and `mp3` are good choices for size and compatibility.

```sh
python3 main.py "path/to/book.epub" "path/to/output/folder" --tts piper --model_name <path_to>/en_US-libritts_r-medium.onnx --piper_speaker 256 --piper_length_scale 1.5 --piper_sentence_silence 0.5 --output_format opus
```

*Alternatively, you can use the following procedure to use piper in a docker container, which simplifies the process of running everything locally.*

1. Ensure you have docker desktop installed on your system. See [Docker](https://www.docker.com/) to install (or use the [homebrew](https://formulae.brew.sh/formula/docker) formula).
2. Download a Piper model & config file (see the [piper repo](https://github.com/rhasspy/piper) for details) and place them in the [piper_models](./piper_models/) directory at the top level of this project.
3. Edit the [docker compose file](./docker-compose.piper-example.yml) to:
   - In the `piper` container, set the `PIPER_VOICE` environment variable to the name of the model file you downloaded.
   - In the `piper` container, map the `volumes` to the location of the piper models on your system (if you used the provided directory described in step 2, you can leave this as is).
   - In the `epub_to_audiobook` container, update the `volumes` mapping from `<path/to/epub/dir/on/host>` to the actual path to the epub on your host machine.
4. From the root of the repo, run `PATH_TO_EPUB_FILE=./Your_epub_file.epub OUTPUT_DIR=$(pwd)/path/to/audiobook_output docker compose -f docker-compose.piper-example.yml up --build`, **replacing the placeholder values and output dirs with your desired epub source and audio output respectively**.  (Leave in the $(pwd) !)  Note that the current config in the docker compose will automatically start the process, entirely in the container. If you want to run the main python process outside the container, you can uncomment the command `command: tail -f /dev/null`, and use `docker exec -it epub_to_audiobook /bin/bash` to connect to the container and run the python script manually (see comments in the  [docker compose file](./docker-compose.piper-example.yml) for more details).

### Examples using Kokoro TTS

The documented usage of Kokoro TTS with this script uses a docker image with endpoints that are OpenAI compatible. However, since it's a "self-hosted" service, you won't need to get an actual key. This requires docker, so follow the docker installation and setup instructions above in the Piper section if you don't have docker on your machine already.

To run, in one terminal tab, run either

```bash
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu
```

Or if you have a GPU that can help with processing, run

```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu
```

Then in another tab, run

```bash
export OPENAI_BASE_URL=http://localhost:8880/v1
export OPENAI_API_KEY="fake"
python main.py path/to/epub output-dir --tts openai --voice_name "af_bella(3)+af_alloy(1)" --model_name "tts-1" #you can replace this with any other voice name. Link below. 
```
Note that passing `--model_name tts-1` parameter **is required** since kokoro breaks with the current default model_name value.

Alternatively, you can do the entire set up through docker compose using the [docker compose file set up for kokoro](./docker-compose.kokoro-example.yml).

To do so, open the file with your favorite editor and then:

- From the root of the repo, run `PATH_TO_EPUB_FILE=./Your_epub_file.epub OUTPUT_DIR=$(pwd)/path/to/audiobook_output VOICE_NAME=Your_desired_voice docker compose -f docker-compose.kokoro-example.yml up --build`, **replacing the placeholder values and output dirs with your desired epub source and audio output respectively, and your voice name**. 
  -  A list of voices can be found [here](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md), and you can sample what they sound like [here](https://huggingface.co/spaces/hexgrad/Kokoro-TTS).
- Note that the current config in the docker compose will automatically start the process, entirely in the container. If you want to run the main python process outside the container, you can uncomment the command `command: tail -f /dev/null`, and use `docker exec -it epub_to_audiobook /bin/bash` to connect to the container and run the python script manually (see comments in the  [docker compose file](./docker-compose.kokoro-example.yml) for more details).


For more information on the image used for kokoro tts, visit this [repo](https://github.com/remsky/Kokoro-FastAPI).

## Troubleshooting

### ModuleNotFoundError: No module named 'importlib_metadata'

This may be because the Python version you are using is [less than 3.8](https://stackoverflow.com/questions/73165636/no-module-named-importlib-metadata). You can try to manually install it by `pip3 install importlib-metadata`, or use a higher Python version.

### FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'

Make sure ffmpeg binary is accessible from your path. If you are on a mac and use homebrew, you can do `brew install ffmpeg`, On Ubuntu you can do `sudo apt install ffmpeg`

### Piper TTS

For installation-related issues, please refer to the [Piper TTS](https://github.com/rhasspy/piper) repository. It's important to note that if you're installing `piper-tts` via pip, [only Python 3.10](https://github.com/rhasspy/piper/issues/509) is currently supported. Mac users may encounter additional challenges when using the downloaded [binary](https://github.com/rhasspy/piper/issues/523). For more information on Mac-specific issues, please check [this issue](https://github.com/rhasspy/piper/issues/395) and [this pull request](https://github.com/rhasspy/piper/pull/412).

Also check [this](https://github.com/p0n1/epub_to_audiobook/issues/85) if you're having trouble with Piper TTS.

## Related Projects

- [Epub to Audiobook (M4B)](https://github.com/duplaja/epub-to-audiobook-hf): Epub to MB4 Audiobook, with StyleTTS2 via HuggingFace Spaces API.
- [Storyteller](https://storyteller-platform.gitlab.io/storyteller/): A self-hosted platform for automatically syncing ebooks and audiobooks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
