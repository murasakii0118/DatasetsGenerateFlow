import os
import subprocess

VIDEO_DIR = "InputVideo"

SCENES_DIR = "Scenes"

OUTPUT_DIR = "videoSegments"

EXECUTION = "step 3.preliminaryFrame\\preliminaryFrame.exe"

PRELIMINARY_FRAME = "preliminaryFrame"

if __name__ == "__main__":
    getDir = [x[1] for x in os.walk(OUTPUT_DIR)][0]
    for dir in getDir:
        try:
            os.mkdir(PRELIMINARY_FRAME+"/"+dir)
        except Exception as e:
            #print(e)
            print(f"{dir} 文件夹已存在,跳过创建")
            pass
        getFile = [x[-1] for x in os.walk(f"{OUTPUT_DIR}/{dir}")][0]
        for name in getFile:
            if(name.find(".txt") >0) :continue
            arg  = [EXECUTION,os.path.abspath(f"{OUTPUT_DIR}/{dir}/{name}"),os.path.abspath(f"./{PRELIMINARY_FRAME}/{dir}")]
            print(arg)
            subprocess.run(arg,start_new_session=True)
        ...
    ...