import yt_dlp

video_link = "https://www.youtube.com/watch?v=GpKgmqV8dWY"

ydl_opts = {
    'format': 'bestvideo[height=2160][fps=60]+bestaudio/best',
    'merge_output_format': 'mp4',
    'outtmpl': 'downloads/%(id)s.%(ext)s',
    'quiet': False,
    'noplaylist': True,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_link])
