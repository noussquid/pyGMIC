import cv2
import sys
import uuid
import subprocess
from queue import Queue 
import threading 
from threading import Thread

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# 3 pipeline process: get image > inpaint image > display image 
# a thread safe producer consumer queue
class ClosableQueue(Queue):
    SENTINEL = object()

    def close(self):
        self.put(self.SENTINEL)

    def __iter__(self):
        while True:
            item = self.get()
            try:
                if item is self.SENTINEL:
                    return 
                yield item
            finally:
                self.task_done()

class StoppableWorker(Thread):

    def __init__(self, func, in_queue, out_queue):
        super(StoppableWorker, self).__init__()
        self.func = func
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        for item in self.in_queue:
            try:
                result = self.func(item)
            except:
                continue

            try:
                self.out_queue.put_nowait(result)
            except:
                continue


def track(frame):
    global tracker 
    print("Tracking frames!")

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw a bounding box
    if ok: 
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0,0,255), -1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking filaure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    #cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(50,170,50),2);

    # Display FPS on Frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(50,170,50),2);

    print("Tracked. Passing on to inpaint")
    track_display.put_nowait(frame)

    return frame


def inpaint(frame):
    print("Inpainting Tracked Image using GMIC")
    unique_filename = str(uuid.uuid4())

    in_file = unique_filename + '_in.png'
    out_file = unique_filename + '_out.png'

    # write frame to image in
    # so gmic can operate on it
    cv2.imwrite(in_file, frame)

    gmic_cmd = "gmic {0} v 0 fx_inpaint_patchmatch 0,9,10,5,1,255,0,0,255,0,0 -o {1}".format(in_file, out_file)
    subprocess.call(gmic_cmd, shell = True)
    
    print("Saved image to: " + out_file + " ready to display")
    return out_file

def captureThreadMain():
    global cam 
    global captureEvt
    global capturing 

    
    
    while True:
        captureEvt.wait()
        if (exiting):
            return

        capturing = True
        captureEvt.clear()
        
        while (capturing and not(exiting)):
            ok, frame = cam.read()
            if not ok:
                print("Failed to grab frame.")
                
            print("Grabbed Frame.")
            print("Passing frame to tracking.")
            try:
                capture_queue.put_nowait(frame)
            except:
                continue

if __name__ == '__main__' :
    global tracker
    global cam
    global capturing
    global exiting
    global captureEvt
    
    capture_queue = ClosableQueue(1)
    track_queue = ClosableQueue(1)
    inpaint_queue = ClosableQueue(1)

    threads = [
        StoppableWorker(track, capture_queue, track_queue),
        StoppableWorker(inpaint, track_queue, inpaint_queue)
    ]
    
    track_display = Queue()

    for thread in threads:
        thread.start()



    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()


    # Try to open camera stream 
    cam = cv2.VideoCapture(0)
    
    # Exit if video not opened 
    if not cam.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame
    ok, frame = cam.read()
    if not ok:
        print "Could nto read first frame"
            
    # Define a bounding box
    bbox = (287, 23, 86, 320)
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box 
    ok = tracker.init(frame, bbox)

    captureThread = threading.Thread(target=captureThreadMain)
    captureThread.daemon = False
    captureEvt = threading.Event()
    exiting = False
    capturing = False

    captureThread.start()
    captureEvt.set()

    while True:
        print("Status of track_display empty: " + str(track_display.empty()))
        if not track_display.empty(): 
            print("Displaying Tracked")
            cv2.imshow("Tracked", track_display.get_nowait())
        
        if not inpaint_queue.empty():
            frame = cv2.imread(inpaint_queue.get_nowait())
            try:
                cv2.imshow("Inpaint", frame)
            except:
                continue

        #Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : 
            exiting = True
            captureEvt.set()
            sys.exit(0)
            break
