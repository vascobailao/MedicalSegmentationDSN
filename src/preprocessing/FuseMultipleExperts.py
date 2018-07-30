import cv2


class FuseMultipleExperts():

    def __init__(self, image, small_cropping_size, big_cropping_size):
        self.small_cropping_size = small_cropping_size
        self.big_cropping_size = big_cropping_size
        self.image = image


    def small_cropping(self):
        image = cv2.imread(self.image)



    def self_consistency_score(self):
        return 0

    def MRF(self):
        return 0