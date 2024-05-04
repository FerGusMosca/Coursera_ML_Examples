import cv2
import numpy as np

from business_entities.image import Image
from common.util.light_logger import LightLogger
from data_access_layer.image_manager import ImageManager


class ImageManagement():

    def __init__(self):
        pass


    @staticmethod
    def __persist__image__(image,settings):#Just in case we need it int he future... but not really

        mgr = ImageManager(settings["ml_reports_conn_str"])
        image.id = mgr.persist_image(image)

        for row in range(0, len(image.pixels)):

            for col in range(0, len(image.pixels[row])):
                image_matrix_id = mgr.persist_image_matrix(row, col, image.id)

                red = int(image.pixels[row][col][0])
                green = int(image.pixels[row][col][1])
                blue = int(image.pixels[row][col][2])

                image_pixel_id = mgr.persist_image_pixel(red, green, blue, image_matrix_id)
                LightLogger.do_log("Persisting pixel id {}: R={} G={} B={}".format(image_pixel_id, red, green, blue))

    @staticmethod
    def extract_image_test(settings):

        path_img = ".\\coursera_course_ML\\input\images\\00000900_022.jpg"

        try:
            # Read the image
            pixels_arr = cv2.imread(path_img)

            image = Image("00000900_022.jpg", "cats", "cats pictures",pixels_arr)

            if(image.pixels is not None):

                LightLogger.do_log("El valor RGB de la imägen  del pixel columna 100, fila 50 es ".format(image.pixels[100][50]))

                # Display the image
                cv2.imshow("Restored Image", image.pixels)
                pass
                #ImageManagement.__persist__image__(image, )

            else:
                LightLogger.do_log("Could not find image for path :{}".format(path_img))
        except Exception as e:
            raise Exception("Critical Error persisting image {}:{}".format(path_img,str(e)))
