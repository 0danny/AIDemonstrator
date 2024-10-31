import subprocess

def main():
    
    print("Making the video...")

    subprocess.run(['python', 'yolov5/detect.py', 
                    '--weights', 'models/best_yolom.pt', 
                    '--source', 'videos/test2.mp4', 
                    '--conf', '0.4', 
                    '--save-txt', 
                    '--save-conf' ])
    
if __name__ == '__main__':
    main()
