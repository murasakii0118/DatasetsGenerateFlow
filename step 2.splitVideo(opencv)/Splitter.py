import sys
import os
import cv2

VIDEO_DIR = "InputVideo"

SCENES_DIR = "Scenes"

OUTPUT_DIR = "videoSegments"
def cutVideo(name : str):
    try:
        ...
        os.mkdir(OUTPUT_DIR+"\\"+name)
    except Exception as e:
        #print(e)
        print(f"{OUTPUT_DIR+'/'+name} 文件夹已存在,跳过创建")
        pass

    fs = open(SCENES_DIR+"\\"+name+".scenes.txt","r")
    lines = fs.readlines()
    cap = cv2.VideoCapture(VIDEO_DIR + "/" + name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{name} 任务开始")
    for index,i in enumerate(lines):
        start_frame,end_frame = i.split(" ")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # type: ignore # 或使用 'XVID' 等编码
        out = cv2.VideoWriter(f"{OUTPUT_DIR}\\{name}\\{index:03}.mp4", fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        frame_count = int(start_frame)
        while frame_count <= int(end_frame):
            ret, frame = cap.read()
            out.write(frame)
            frame_count += 1
        out.release()
        print(f"第{index}段已完成")

    cap.release()
    
    fs.close()
    ...

if __name__ == "__main__":
    getVideo = [x[-1] for x in os.walk("InputVideo")][0]
    for i in getVideo:
        cutVideo(i)
    cv2.destroyAllWindows()
    _dir = [x[0] for x in os.walk(f"{OUTPUT_DIR}")][1:]
    print(_dir)
    for idx,j in enumerate(_dir):
        print(f"{j},{OUTPUT_DIR}\\{idx:04d}")
        os.rename(f"{j}\\",f"{OUTPUT_DIR}\\{idx:04d}")
        name = j.split('\\')[-1]
        open(f"{OUTPUT_DIR}\\{idx:04d}\\{name}.txt","w").close()