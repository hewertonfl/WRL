import numpy as np
import cv2
#imgToSeg = Perspective.do(img_numpy0, centroFuros)
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect
	
def four_points_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = (max(int(widthA), int(widthB))+00)
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = (max(int(heightA), int(heightB))+00)
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [+300, +300],
        [maxWidth + 300, +300],
        [maxWidth + 300, maxHeight + 300],
        [300, maxHeight + 300]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    
    warped = cv2.warpPerspective(image, M, (maxWidth+800, maxHeight+800))
    # return the warped image
    return warped
	
def six_points_transform(image, pts):
    # Read source image. (Imagem a ser transformada)
    im_src = image
    # Source image's points
    pts_src = pts

    # Read destination image. (Imagem de modelo para a transformação)
    im_dst = cv2.imread('hex.jpg')
    # Destination images's points.
    pts_dst = np.array([[1260,551],[1115,803],[821,803],[674,551],[821,299],[1115,299]]) 

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    return im_out
    
def five_points_transform(image, pts):
    # Read source image. (Imagem a ser transformada)
    im_src = image
    # Source image's points
    pts_src = pts

    # Read destination image. (Imagem de modelo para a transformação)
    im_dst = cv2.imread('pen.png')
    # Destination images's points.
    pts_dst = np.array([[643, 432],[965,199],[1286,432],[1163,810],[767,810]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    return im_out
	
def do(image, centros):
    if(len(centros)==6):
        pts = np.float32([(centros[0]), (centros[1]), (centros[2]), (centros[3]), (centros[4]), (centros[5])])
        # apply the four points tranform to obtain a "birds eye view" of the image   
        warped = six_points_transform(image, pts)
    elif(len(centros)==4):
        pts = np.float32([(centros[1]), (centros[0]), (centros[2]), (centros[3])])
        # apply the four points tranform to obtain a "birds eye view" of the image   
        warped = four_points_transform(image, pts)
    elif(len(centros)==5):
        pts = np.float32([(centros[0]), (centros[1]), (centros[2]), (centros[3]), (centros[4])])
        # apply the four points tranform to obtain a "birds eye view" of the image   
        warped = five_points_transform(image, pts)
    else:
        # apply the four point tranform to obtain a "birds eye view" of the image   
        warped = image
    
    
    #cv2.imshow("Original", image)
    
    return(warped)
