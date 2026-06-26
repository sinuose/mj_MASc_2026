# author martin
# date \today
import numpy as np
import mat73
import cv2



def load_video(path_to_video):
    '''
    video loader for matlab data (H,W,F), mp4, and tif (WIP)
    :param path_to_video:
    :return: video_array
    '''

    if path_to_video.split('.')[-1] == 'mat':
        try:
            video_array = mat73.loadmat(path_to_video)
            img = video_array['Im_stack']
            #H, W, F = img.shape
            video_array = np.transpose(img, (2, 0, 1))  # shape (F, H, W)
            return video_array
        except Exception as e:
            return 0

    elif path_to_video.split('.')[-1] == 'mp4':
        cap = cv2.VideoCapture(path_to_video)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads in BGR format, convert to RGB if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()

        # # Convert to numpy array
        video_array = np.array(frames)
        return video_array

    elif path_to_video.split('.')[-1] == 'nd2':
        import nd2

        try:
            nd2_data = nd2.imread(path_to_video)
            #print(f"Loaded {file_path} with shape {np.shape(data)}")
            return nd2_data
        except Exception as e:
            print(f"Error loading {path_to_video}: {e}")
            return 0

