import cv2
import numpy as np

sudoku = cv2.imread("sudoku-original.jpg")

gray_sudoku = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_sudoku, (9,9), 0)

original_threshold = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)

kernel = np.uint8([[0,1,0],[1,1,1],[0,1,0]])
original_threshold = cv2.dilate(original_threshold, kernel=kernel)

contours, hierachy = cv2.findContours(original_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

biggest_blob = max(contours, key=cv2.contourArea)

peri = cv2.arcLength(biggest_blob, True)
approx = cv2.approxPolyDP(biggest_blob, 0.05*peri, True)

list_corner = []
for corner in approx:
    list_corner.append(corner)
list_corner = [[[373, 52]],[[54, 63]],[[25,387]],[[392, 393]]]
input_pnt = np.float32(list_corner)
output_pnt = np.float32([[sudoku.shape[0], 0], [0,0], [0, sudoku.shape[1]], [sudoku.shape[0], sudoku.shape[1]]])
new_location = cv2.getPerspectiveTransform(input_pnt, output_pnt)
focus_sudoku = cv2.warpPerspective(gray_sudoku, new_location, (sudoku.shape[1], sudoku.shape[0]), flags=cv2.INTER_LINEAR)

focus_blur = cv2.GaussianBlur(focus_sudoku, (9,9), 0)
focus_threshold = cv2.adaptiveThreshold(focus_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 4)
cv2.imwrite("output.jpg",focus_threshold)

column_kernel = np.uint8([[0,0,0],[1,1,1],[0,0,0]])
row_kernel = np.uint8([[0,1,0],[0,1,0],[0,1,0]])

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
detect_horizontal = cv2.morphologyEx(focus_threshold, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

for i in range(70):
    detect_horizontal = cv2.dilate(detect_horizontal, kernel=horizontal_kernel)

for i in range(1):
    detect_horizontal = cv2.erode(detect_horizontal, kernel=row_kernel)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
detect_vertical = cv2.morphologyEx(focus_threshold, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

for i in range(50):
    detect_vertical=cv2.dilate(detect_vertical, kernel=vertical_kernel)


full_mask = cv2.bitwise_or(detect_horizontal, detect_vertical)
position = cv2.bitwise_and(detect_horizontal, detect_vertical)


contours, hierachy = cv2.findContours(position, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
x_position = []
y_position = []
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    x_position.append(x)
    y_position.append(y)

x_position.sort()
y_position.sort()

x_position = list(dict.fromkeys(x_position))
y_position = list(dict.fromkeys(y_position))

focus_kernel = np.uint8([[0,1,0],[0,1,0],[0,1,0]])
focus_dilate = cv2.dilate(focus_threshold, kernel=focus_kernel)

focus_contours, hierachy = cv2.findContours(focus_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

number_position_x = []
number_position_y = []

for contour in focus_contours:
    x,y,w,h = cv2.boundingRect(contour)
    if w < 30 and w > 10 and h > 20:
        number_position_x.append(x)
        number_position_y.append(y)

number_map = [[0,0,0,0,0,0,0,0,0], 
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0]]

for i in range(len(number_position_x)):
    x = 0
    y = 0
    for j in range(len(x_position)):
        if number_position_x[i] < x_position[j]:
            x = j - 1
            break
    for z in range(len(y_position)):
        if number_position_y[i] < y_position[z]:
            y = z - 1
            break
    number_map[y][x] = 1

number_map = np.array(number_map)

result_string = ""

for i in range(len(number_map)):
    for j in number_map[i]:
        if j == 0:
            result_string += " "
        else:
            result_string += "X"
    result_string += "\n"

text_file = open("output.txt", "w")
n = text_file.write(result_string)
text_file.close()
