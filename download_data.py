import pandas as pd
import requests
import concurrent.futures
import os

def download_video(row):
    video_url = row['contentUrl']  
    video_name = row['videoid']  
    folder_path = './datasets_train'  # 请将此路径替换为你的文件夹的路径
    video_file = os.path.join(folder_path, f'{video_name}.mp4')

    # 如果文件已经存在，就跳过下载
    if os.path.isfile(video_file):
        return

    response = requests.get(video_url)

    if response.status_code == 200:
        with open(video_file, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download video {video_name} from url {video_url}")

df = pd.read_csv('results_2M_train.csv')

rows = df.to_dict('records')

with concurrent.futures.ThreadPoolExecutor() as executor:
    for row in rows:
        executor.submit(download_video, row)