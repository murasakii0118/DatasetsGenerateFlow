import os
import subprocess

VIDEO_DIR = "InputVideo"

SCENES_DIR = "Scenes"

OUTPUT_DIR = "videoSegments"

EXECUTION = "step 4.precisionFrame\\classify.exe"

PRELIMINARY_FRAME = "preliminaryFrame"

PRECISION_FRAME = "precisionFrame"

if __name__ == "__main__":
    getDir = [x[1] for x in os.walk(PRELIMINARY_FRAME)][0]
    for dir in getDir:
        try:
            os.mkdir(PRECISION_FRAME+"/"+dir)
        except Exception as e:
            #print(e)
            print(f"{dir} 文件夹已存在,跳过创建")
            pass
        getFile = [x[-1] for x in os.walk(f"{PRELIMINARY_FRAME}/{dir}/general")][0]
        arg  = [EXECUTION,os.path.abspath(f"{PRELIMINARY_FRAME}/{dir}/general/"),os.path.abspath(f"./{PRECISION_FRAME}/{dir}")]
        print(arg)
        subprocess.run(arg,start_new_session=True)
    ...