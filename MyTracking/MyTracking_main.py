from ast import Pass
from glob import glob
from turtle import back
from matplotlib import image
import mmcv
import cv2
import tempfile
import matplotlib.pyplot as plt
from regex import B
from mmtrack.apis import inference_sot, init_model
import tkinter as tk
from tkinter import filedialog
import time
from PIL import Image, ImageTk, ImageSequence
import random

#视频目标追踪
def Video_Tracking(input_video, output, box, num, mode=False):
    # 指定单目标追踪算法 config 配置文件
    sot_config = './configs/sot/stark/stark_st1_r50_500e_lasot.py'
    # 指定单目标检测算法的模型权重文件
    sot_checkpoint = 'https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220223_125402-934f290e.pth'
    # 初始化单目标追踪模型
    sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')
    # 指定初始框的坐标 [x, y, w, h]
    init_bbox = list(box)
    # 转成 [x1, y1, x2, y2 ]
    init_bbox = [init_bbox[0], init_bbox[1], init_bbox[0]+init_bbox[2], init_bbox[1]+init_bbox[3]]

    # 读入待预测视频
    imgs = mmcv.VideoReader(input_video)
    fps = imgs.fps
    imgs = imgs[num:]
    prog_bar = mmcv.ProgressBar(len(imgs))
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name
    trace_coord_video = []
    # 逐帧输入模型预测
    for i, img in enumerate(imgs):
        result = inference_sot(sot_model, img, init_bbox, frame_id=i)
        # 绘制矩形框中心点构成的轨迹
        result_int = list(result['track_bboxes'].astype('uint32'))
        trace_coord_video.append(result_int)
        if mode == True:
            for trace_coord_obj in trace_coord_video: # 遍历每一帧
                cv2.circle(img, (int((trace_coord_obj[0]+trace_coord_obj[2])/2), int((trace_coord_obj[1]+trace_coord_obj[3])/2)),5, (255,0,0), -1) 
        #等待键入数据
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        frame = sot_model.show_result(
                img,
                result,
                wait_time=int(1000. / fps),
                out_file=f'{out_path}/{i:06d}.jpg')
        cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_KEEPRATIO)
        cv2.imshow('frame', frame)
        prog_bar.update()
    print(f'\n making the output video at {output} with a FPS of {fps}')
    if len(output) > 0:
        mmcv.frames2video(out_path, output, fps=fps, fourcc='mp4v')
    out_dir.cleanup()
    cv2.destroyAllWindows()

def get_fps_image(input_path, num):
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, num)  #设置要获取的帧号
    if cap.isOpened():
        success, frame = cap.read()
    cap.release()
    # 选取初始框区域，按回车键完成
    target = cv2.selectROI("initial bbox", frame)
    return target

def play_video(path):
    cap = cv2.VideoCapture(path)
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算每帧显示时长，单位ms
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wait = int(1 / fps * 750)
    cv2.namedWindow("frame", 0)
    cv2.createTrackbar('time', 'frame', 0, frames, Pass)
    loop_flag = 0
    pos = 0
    while True:
        if loop_flag == pos:
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', 'frame', loop_flag)
        else:
            pos = cv2.getTrackbarPos('time', 'frame')
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        # waitKey指定每帧显示时长，单位为毫秒
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        time.sleep(int(1 / fps))
    cap.release()
    cv2.destroyAllWindows()
    return(loop_flag)

def get_webcam_image(input_path):
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        success, frame = cap.read()
    cap.release()
    # 选取初始框区域，按回车键完成
    target = cv2.selectROI("initial bbox", frame)
    return target

def Webcam_Tracking(input_video, output, box, mode=False):
    # 指定单目标追踪算法 config 配置文件
    sot_config = './configs/sot/stark/stark_st1_r50_500e_lasot.py'
    # 指定单目标检测算法的模型权重文件
    sot_checkpoint = 'https://download.openmmlab.com/mmtracking/sot/stark/stark_st1_r50_500e_lasot/stark_st1_r50_500e_lasot_20220223_125402-934f290e.pth'
    # 初始化单目标追踪模型
    sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')
    # 指定初始框的坐标 [x, y, w, h]
    init_bbox = list(box)
    # 转成 [x1, y1, x2, y2 ]
    init_bbox = [init_bbox[0], init_bbox[1], init_bbox[0]+init_bbox[2], init_bbox[1]+init_bbox[3]]
    # 读入待预测视频
    camera = cv2.VideoCapture(input_video)
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name
    trace_coord_video = []
    i=0
    # 逐帧输入模型预测
    while True:
        ret_val, img = camera.read()
        result = inference_sot(sot_model, img, init_bbox, frame_id=i)
        # 绘制矩形框中心点构成的轨迹
        result_int = list(result['track_bboxes'].astype('uint32'))
        trace_coord_video.append(result_int)
        if mode == True:
            for trace_coord_obj in trace_coord_video: # 遍历每一帧
                cv2.circle(img, (int((trace_coord_obj[0]+trace_coord_obj[2])/2), int((trace_coord_obj[1]+trace_coord_obj[3])/2)),5, (255,0,0), -1)
        frame = sot_model.show_result(
                img,
                result,
                wait_time=0,
                out_file=f'{out_path}/{i:06d}.jpg')
        #等待键入数据
        cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_KEEPRATIO)
        cv2.imshow('frame', frame)
        i+=1
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            mmcv.frames2video(out_path, output, fps=30, fourcc='mp4v')
            out_dir.cleanup()
            cv2.destroyAllWindows()
            break

def pick(event):
    while 1:
        im = Image.open(gif_address+gif_list[gif_num]+'.gif')
        # GIF图片流的迭代器
        iter = ImageSequence.Iterator(im)
        #frame就是gif的每一帧，转换一下格式就能显示了
        for frame in iter:
            #将frame放大至窗口大小
            frame = frame.resize((window_width,window_height),Image.ANTIALIAS)
            pic=ImageTk.PhotoImage(frame)
            canvas.create_image(window_width/2,window_height/2,image=pic)
            time.sleep(0.1)
            root.update_idletasks()  #刷新
            root.update()
def B1():
    tk.messagebox.showinfo( "MyTracking", "如有疑问请联系E-mail: 1952127@tongji.edu.cn")
def B2():
    tk.messagebox.showinfo( "MyTracking", "选择你想使用的视频")
    root1 = tk.Tk()
    root1.title('Choose the video')
    #选择视频文件
    root1.withdraw()
    f_path = filedialog.askopenfilename()
    #获取视频片段
    fps_num = play_video(f_path)
    #获取目标区域
    target = get_fps_image(f_path, fps_num)
    tk.messagebox.showinfo( "MyTracking", "选择视频保存位置")
    #选择保存位置
    s_path = filedialog.asksaveasfilename()
    #调用单目标追踪
    Video_Tracking(f_path, s_path, target, fps_num)
    root1.destroy()

def B3():
    tk.messagebox.showinfo( "MyTracking", "请打开摄像头")
    root2 = tk.Tk()
    root2.title('Select the location to save the video')
    root2.withdraw()
    tk.messagebox.showinfo( "MyTracking", "选择视频保存位置")
    s_path = filedialog.asksaveasfilename()
    target = get_webcam_image(0)
    #选择保存位置
    Webcam_Tracking(0, s_path, target)
    root2.destroy()

def B4():
    tk.messagebox.showinfo( "MyTracking", "选择你想使用的视频")
    root3 = tk.Tk()
    root3.title('Choose the video')
    #选择视频文件
    root3.withdraw()
    f_path = filedialog.askopenfilename()
    #获取视频片段
    fps_num = play_video(f_path)
    #获取目标区域
    target = get_fps_image(f_path, fps_num)
    tk.messagebox.showinfo( "MyTracking", "选择视频保存位置")
    #选择保存位置
    s_path = filedialog.asksaveasfilename()
    #调用单目标追踪
    Video_Tracking(f_path, s_path, target, fps_num, True)
    root3.destroy()

def B5():
    tk.messagebox.showinfo( "MyTracking", "请打开摄像头")
    root4 = tk.Tk()
    root4.title('Select the location to save the video')
    root4.withdraw()
    tk.messagebox.showinfo( "MyTracking", "选择视频保存位置")
    s_path = filedialog.asksaveasfilename()
    target = get_webcam_image(0)
    #选择保存位置
    Webcam_Tracking(0, s_path, target, True)
    root4.destroy()

def on_closing():
    if tk.messagebox.askokcancel("MyTracking", "是否退出窗口？"):
        root.destroy()


#初始化窗体
window_width = 900
window_height = 600
root = tk.Tk()
root.geometry(str(window_width)+'x'+str(window_height))
root.title('MyTracking')
root.configure(background='#DEEBF7') 
root.resizable(False, False)
B1 = tk.Button(root, text ="关于MyTracking", command = B1, height = 1, width = 20, bg = '#DEEBF7', relief='groove')
B2 = tk.Button(root, text ="选择视频", command = B2, height = 1, width = 8, bg = '#DEEBF7')
B3 = tk.Button(root, text ="调用摄像头", command = B3, height = 1, width = 8, bg = '#DEEBF7')
B4 = tk.Button(root, text ="视频轨迹", command = B4, height = 1, width = 8, bg = '#DEEBF7')
B5 = tk.Button(root, text ="摄像头轨迹", command = B5, height = 1, width = 8, bg = '#DEEBF7')
B1.pack()
B2.pack()
B2.place(x=0, y=0)
B3.pack()
B3.place(x=68, y=0)
B4.pack()
B4.place(x=136, y=0)
B5.pack()
B5.place(x=204, y=0)

canvas = tk.Canvas(root,width=window_width, height=window_height,bg='#DEEBF7')
canvas.pack()
gif_list = ['大礼堂','国立柱','瑞安楼','三好坞','图书馆','文远楼','校门','樱花大道','综合楼']
gif_num = random.randint(0,8)
gif_address = './data/GIF/'

canvas.bind("<Enter>",pick)
#退出部分
root.protocol('WM_DELETE_WINDOW', on_closing)
root.mainloop()


