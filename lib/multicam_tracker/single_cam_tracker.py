from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence
from deeptam_tracker.tracker import Tracker

class SingleCamTracker:

    # TODO: require_depth, require_pose
    def __init__(self, config, tracking_module_path, checkpoint, require_depth=False, require_pose=False):
    
        self.sequence = RGBDSequence(config['cam_dir'], rgb_parameters=config['rgb_parameters'],
                            depth_parameters=config['depth_parameters'],
                            time_syncing_parameters=config['time_syncing_parameters'])
        
        self.intrinsics = self.sequence.get_original_normalized_intrinsics()

        self.tracker = Tracker(tracking_module_path,
                          checkpoint,
                          self.intrinsics,
                          tracking_parameters=config['tracking_parameters'])
        

    def startup():
        frame = self.sequence.get_dict(0, 
                                       self.intrinsics, 
                                       self.tracker.image_width, 
                                       self.tracker.image_height)

        pose0_gt = frame['pose']
        
        self.tracker.clear()
        # WIP: If gt_poses is aligned such that it starts from identity pose, you may comment this line
        # TODO: @Rohit, should we make this base-to-cam transformation?
        self.tracker.set_init_pose(pose0_gt)
    
    def update(self, frame_idx):
        frame = self.sequence.get_dict(frame_idx, 
                                        self.intrinsics, 
                                        self.tracker.image_width, 
                                        self.tracker.image_height)

        result = self.tracker.feed_frame(frame['image'], frame['depth'])
        
        return frame, result, self.sequence.get_timestamp(frame_idx), self.tracker.poses 
    
    def get_sequence_length(self):
        return self.sequence.get_sequence_length()

    def delete_tracker(self):
        del self.tracker