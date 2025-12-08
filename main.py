from visualizer import DataLoader, Visualizer
from ekf_slam import EKFSlam

if __name__ == '__main__':
    history_file = "output/slam_history.json"
    
    sim = EKFSlam()
    sim.simulate()
    sim.save(output_file=history_file)
    
    dl = DataLoader(file_path=history_file)
    viz = Visualizer(dl)
    
    print("Visualizer controls: space=step, r=run/pause, q=quit")
    viz.run()
    