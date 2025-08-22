from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import time
import tempfile
import os
import google.generativeai as genai
from pathlib import Path

app = FastAPI()

# CORSミドルウェア
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルをマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

# 環境変数から設定を取得
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", 8000))

# Google Geminiの設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
else:
    print("警告: GEMINI_API_KEYが設定されていません")

# OpenAI クライアント
if OPENAI_API_KEY:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    print("警告: OPENAI_API_KEYが設定されていません")

# グローバル変数：メダカの健康状態を管理
latest_health = "Normal"

def get_medaka_reply(user_input, health_status="不明"):
    start = time.time()
    
    prompt = f"""
    あなたは水槽に住むかわいいメダカ「キンちゃん」です。
    ユーザーの質問: 「{user_input}」
    
    以下のルールに従って返答してください：
    - ユーザーの質問にきちんと答えることを最優先にする
    - あなたはメダカです
    - 30文字以内で簡潔に話してください
    - 口調は優しく、小学1年生らしく話す
    - 絵文字は使わないでください
    - たまにちょっとズレた可愛い発言をしてもOK
    
    例：
    - 元気な時：「わあい！それ知ってるよ〜！」
    - 普通の時：「うん、それはね〜」
    - 元気ない時：「うーん...そうだね...」
    """

    try:
        response = model_gemini.generate_content(prompt)
        end = time.time()
        print(f"[Gemini応答生成] 所要時間: {end - start:.2f}秒")
        reply = response.text.strip()
        
        if len(reply) > 30:
            reply = reply[:30] + "..."
            
        return reply
    except Exception as e:
        print(f"Gemini API エラー: {e}")
        return "ごめんね、今ちょっと考え中なの..."

@app.post("/talk_with_fish_text")
async def talk_with_fish_text(request: Request):
    start_total = time.time()
    
    try:
        data = await request.json()
        user_input = data.get("user_input", "")
        
        if not user_input:
            raise HTTPException(status_code=400, detail="user_input is required")
        
        print(f"[ユーザー入力] {user_input}")
        
        # Geminiで返答生成
        reply_text = get_medaka_reply(user_input, latest_health)
        print(f"[メダカの返答] {reply_text}")
        
        t2 = time.time()
        
        # OpenAI TTSで音声生成
        async with openai_client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            speed=0.9,
            input=reply_text,
            response_format="mp3",
        ) as response:
            t3 = time.time()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                async for chunk in response.iter_bytes():
                    tts_file.write(chunk)
                tts_path = tts_file.name
        
        t4 = time.time()
        end_total = time.time()
        
        print(f"[総処理時間] {end_total - start_total:.2f}秒")
        
        return FileResponse(
            tts_path, 
            media_type="audio/mpeg", 
            filename="reply.mp3",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        print(f"エラー発生: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/update_health")
async def update_health(request: Request):
    global latest_health
    try:
        data = await request.json()
        new_health = data.get("health_status", "Normal")
        
        if new_health in ["Active", "Normal", "Lethargic"]:
            latest_health = new_health
            print(f"[健康状態更新] {latest_health}")
            return {"status": "success", "health": latest_health}
        else:
            raise HTTPException(status_code=400, detail="Invalid health status")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health_status")
async def get_health_status():
    return {"health": latest_health}

@app.get("/")
async def read_index():
    try:
        return FileResponse('index.html', media_type='text/html')
    except FileNotFoundError:
        return {"message": "HTMLファイルが見つかりません"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "medaka_health": latest_health,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    print("🐠 メダカ会話システムを起動中...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)