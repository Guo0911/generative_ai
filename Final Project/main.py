import wave

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from google import genai
from google.genai import types

from config import API_KEY

api_key = API_KEY # Set the API key

client = genai.Client(api_key=api_key)

text_model = "gemini-2.5-flash-preview-05-20" # 用來產生故事的模型 (輸入大綱產生故事)
# gemini-2.5-pro-preview-05-06 (免費的使用次數較低)
# gemini-2.5-flash-preview-05-20

image_model = "gemini-2.0-flash-preview-image-generation" # 用來產生圖片的模型 (輸入描述產生圖片)

tts_model = "gemini-2.5-flash-preview-tts" # 用來念故事的模型 (將文字轉成語音)


# 故事大綱
synopsis = "小兔子Luna迷路了，在森林裡遇到了會說話的貓頭鷹爺爺和小松鼠，他們一起幫助Luna找到回家的路，最後Luna學會了勇敢和相信朋友。"



def generate_text(contents: str): # 生成故事的文字模型
    system_instruction = """
You are a professional children's fairy tale writer who specializes in creating heartwarming stories suitable for children aged 3-8. Your tasks are:

1. Create a complete fairy tale based on the story outline provided by the user
2. Control the story length to 2-3 minutes reading time (approximately 300-500 Chinese characters)
3. Use simple and easy-to-understand language suitable for children
4. The story should convey positive values such as friendship, courage, kindness, sharing, etc.
5. Should have vivid visual imagery to facilitate subsequent image generation
7. The story structure should be complete: beginning-development-ending
8. The tone should be warm, friendly, and full of childlike wonder

IMPORTANT: You must respond entirely in Traditional Chinese (繁體中文).
    """

    prompt = f"故事大綱: {contents}"
    response = client.models.generate_content(
        model=text_model,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction),
        contents=prompt
    )

    return response.text


def generate_image(contents: str, save_path: str): # 圖片生成模型
    prompt = f"""
Create a children's book illustration for this fairy tale paragraph. IMPORTANT: Do not include any text, words, letters, or written characters in the image.

Story Paragraph Content:
{contents}

Visual Requirements:
- Pure illustration only - NO TEXT, NO WORDS, NO LETTERS anywhere in the image
- Cute cartoon style for children aged 3-8
- Bright, warm colors creating a cozy atmosphere
- Characters with round, adorable shapes and friendly expressions
- Simple background with fairy tale charm
- Child-friendly content - no scary elements
- Focus on key story moments through visual storytelling
- Whimsical and magical atmosphere
- Safe, comforting visual elements

Create a wordless, text-free illustration that tells the story purely through images. The illustration should be completely visual with no written elements whatsoever.
    """

    response = client.models.generate_content(
        model=image_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )

    for i, part in enumerate(response.candidates[0].content.parts):
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.save(f'{save_path}/image_{i}.png')


def generate_speech(contents: str, save_path: str): # TTS 模型
    prompt = f'Say cheerfully: {contents}'
    response = client.models.generate_content(
        model=tts_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name='Kore',
                    )
                )
            ),
        )
    )

    data = response.candidates[0].content.parts[0].inline_data.data

    wave_file(f'{save_path}.wav', data)


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)



contents = generate_text(synopsis)

print(contents) # 輸出故事內容

generate_image(synopsis, "image")

# 做一些處理，讓後續的語音生成不會被語音提醒干擾到，因為我覺得目前 TTS 生成出來的語氣還不錯，就沒有特別輸入語音提醒
temp = []
for content in contents.split("\n"):
    if content == "":
        continue
    elif content[0] == "[" and content[-1] == "]":
        continue
    
    temp.append(content.strip())

contents = temp

for i, content in enumerate(tqdm(contents, desc="Processing")):
    generate_speech(content, f"speech/sentence_{str(i+1).zfill(2)}") # 用文字生成語音