# Libraries required for code
import numpy as np
import cv2
#import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Algorithm Functions
#-----------------------------------------------------------------------------

def read_image(img_name: str):
    img = cv2.imread(img_name)

    if img.size != 0:
        return img
    else:
        print("An error has occurred when reading in the image")
        return None

def get_contours(image):
    contours,_ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_contour_area(contours):
    cnt_area = []

    for i in range(0,len(contours),1):
        cnt_area.append(cv2.contourArea(contours[i]))

    list.sort(cnt_area,reverse=True)
    return cnt_area

def draw_bounding_box(contours,image, number_of_boxes=1):
    cnt_area = find_contour_area(contours)

    for i in range(0,len(contours),1):
        cnt = contours[i]

        if (cv2.contourArea(cnt) > cnt_area[number_of_boxes]):
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),10)
            #print(x,y,w,h)

    return x,y,w,h

def crop_to_image(image,x,y,w,h):
    cropped_image = image[y:y+h,x:x+w,:]
    return cropped_image

def generate_masks(input_image):
    hls_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2HLS)
    
    lower_blue = np.array([90,0,0])
    upper_blue = np.array([100,255,255])

    lower_green = np.array([35,0,0])
    upper_green = np.array([45,255,255])

    lower_yellow = np.array([20,0,0])
    upper_yellow = np.array([23,255,255])

    lower_red = np.array([0,0,0])
    upper_red = np.array([5,255,255])

    red_mask = cv2.inRange(hls_image,lower_red,upper_red)
    green_mask = cv2.inRange(hls_image,lower_green,upper_green)
    yellow_mask = cv2.inRange(hls_image,lower_yellow,upper_yellow)
    blue_mask = cv2.inRange(hls_image,lower_blue,upper_blue)

    return red_mask, green_mask, blue_mask, yellow_mask

def determineCardColour(red_mask, green_mask, yellow_mask, blue_mask):
    red_mask_probs = (red_mask == 255).sum()/red_mask.size
    green_mask_probs = (green_mask == 255).sum()/green_mask.size
    yellow_mask_probs = (yellow_mask == 255).sum()/yellow_mask.size
    blue_mask_probs = (blue_mask == 255).sum()/blue_mask.size

    mask_probs = np.array([red_mask_probs,green_mask_probs,yellow_mask_probs,blue_mask_probs])

    max_mask_prob_loc = np.argmax(mask_probs)
    if (max_mask_prob_loc == 0):
        return "Red", mask_probs[0]
    elif (max_mask_prob_loc == 1):
        return "Green", mask_probs[1]
    elif (max_mask_prob_loc == 2):
        return "Yellow", mask_probs[2]
    elif (max_mask_prob_loc == 3):
        return "Blue", mask_probs[3]
    else:
        return "Error occurred", 0
    

def locate_card_in_image(colour_image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(colour_image,cv2.COLOR_BGR2GRAY)
    # Convert to binary
    binary_image = np.ones_like(gray_image)
    cv2.threshold(gray_image,180,255,cv2.THRESH_BINARY,binary_image)
    # Find contours
    contours = get_contours(binary_image)

    # Draw bounding box from contours
    image_copy = np.copy(colour_image)
    x,y,w,h = draw_bounding_box(contours,image_copy)

    # Extract detected region from image
    cropped_image = crop_to_image(image_copy,x,y,w,h)

    # Generates binary masks for the detection of the 4 main card colours
    red_mask, green_mask, blue_mask, yellow_mask = generate_masks(cropped_image)

    # Determine the card colour from the largest amount of colour pixels in masks
    colour, colourProb = determineCardColour(red_mask,green_mask,yellow_mask,blue_mask)

    # Draw output card class and associated class probability
    text_to_display = colour + " - " + str(round(colourProb,4))
    text_x = x
    text_y = y - 15
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text_to_display,font,4,10)
    cv2.rectangle(image_copy,(text_x, text_y - text_height - baseline),(text_x + text_width, text_y + baseline),(255,255,255),cv2.FILLED)
    cv2.putText(image_copy,text_to_display,(text_x,text_y),font,4,(0,0,0),10)
    
    return gray_image, binary_image, cropped_image, image_copy, red_mask, green_mask, yellow_mask, blue_mask


#--------------------------------------------------------------
# Main Code
#--------------------------------------------------------------

# read in image


input_image = read_image("bluedemonslayercard.jpg")

gray_image = np.ones_like(input_image)
binary_image = np.ones_like(gray_image)
output_image = np.ones_like(input_image)

# display colour, grayscale, binary, mask, output image
gray_image, binary_image, cropped_image, output_image,_,_,_,_ = locate_card_in_image(input_image)

while True:

    cv2.namedWindow("Input Image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Input Image",input_image)
    cv2.resizeWindow('Input Image', int(input_image.shape[1]/4), int(input_image.shape[0]/4))

    cv2.namedWindow("Gray Image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Gray Image",gray_image)
    cv2.resizeWindow('Gray Image', int(gray_image.shape[1]/4), int(gray_image.shape[0]/4))

    cv2.namedWindow("Cropped Image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Cropped Image",cropped_image)
    cv2.resizeWindow('Cropped Image', int(cropped_image.shape[1]/4), int(cropped_image.shape[0]/4))

    cv2.namedWindow("Output Image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Output Image",output_image)
    cv2.resizeWindow('Output Image', int(output_image.shape[1]/4), int(output_image.shape[0]/4))

    # Hit Escape Key to Close the Windows for now
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()