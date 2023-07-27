import cv2 as cv
import mediapipe as mp
import copy
from collections import deque, Counter
import csv
import numpy as np
import itertools
import time
import pandas as pd

from model import KeyPointClassifier
# from playsound import playsound


class CvFpsCalc(object):
	def __init__(self, buffer_len=1):
		self._start_tick = cv.getTickCount()
		self._freq = 1000.0 / cv.getTickFrequency()
		self._difftimes = deque(maxlen=buffer_len)

	def get(self):
		current_tick = cv.getTickCount()
		different_time = (current_tick - self._start_tick) * self._freq
		self._start_tick = current_tick
		self._difftimes.append(different_time)
		fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
		fps_rounded = round(fps, 2)
		return fps_rounded


def calc_bounding_rect(image, landmarks):
	# return rect around hand
	image_width, image_height = image.shape[1], image.shape[0]
	landmark_array = np.empty((0, 2), int)
	for _, landmark in enumerate(landmarks.landmark):
		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)
		landmark_point = [np.array((landmark_x, landmark_y))]
		landmark_array = np.append(landmark_array, landmark_point, axis=0)
	x, y, w, h = cv.boundingRect(landmark_array)
	return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
	image_width, image_height = image.shape[1], image.shape[0]
	landmark_point = []
	# Keypoint
	for _, landmark in enumerate(landmarks.landmark):
		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)
		landmark_point.append([landmark_x, landmark_y])
	return landmark_point


def pre_process_landmark(landmark_list):
	temp_landmark_list = copy.deepcopy(landmark_list)
	# Convert to relative coordinates
	base_x, base_y = 0, 0
	for index, landmark_point in enumerate(temp_landmark_list):
		if index == 0:
			base_x, base_y = landmark_point[0], landmark_point[1]
		temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
		temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
	# Convert to a one-dimensional list
	temp_landmark_list = list(
		itertools.chain.from_iterable(temp_landmark_list))
	# Normalization
	max_value = max(list(map(abs, temp_landmark_list)))
	def normalize_(n):
		return n / max_value
	temp_landmark_list = list(map(normalize_, temp_landmark_list))
	return temp_landmark_list


def draw_info_text(image, brect, label, hand_sign_text, color1=(0, 0, 0), color2=(255, 255, 255)):
	cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), color1, -1)
	info_text = label
	if hand_sign_text != "":
		info_text = info_text + ':' + hand_sign_text
	cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, color2, 1, cv.LINE_AA)
	return image


def draw_point_history(image, point_history, color=(152, 251, 152)):
	for index, point in enumerate(point_history):
		if point[0] != 0 and point[1] != 0:
			cv.circle(image, (point[0], point[1]), 1 + int(index / 2), color, 2)
	return image


def draw_landmarks(image, landmark_point, color1=(0, 0, 0), color2=(255, 255, 255)):
	if len(landmark_point) > 0:
		# Thumb
		cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), color1, 6)
		cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), color2, 2)
		cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), color1, 6)
		cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), color2, 2)

		# Index finger
		cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), color1, 6)
		cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), color2, 2)
		cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), color1, 6)
		cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), color2, 2)
		cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), color1, 6)
		cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), color2, 2)

		# Middle finger
		cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), color1, 6)
		cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), color2, 2)
		cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), color1, 6)
		cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), color2, 2)
		cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), color1, 6)
		cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), color2, 2)

		# Ring finger
		cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), color1, 6)
		cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), color2, 2)
		cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), color1, 6)
		cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), color2, 2)
		cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), color1, 6)
		cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), color2, 2)

		# Little finger
		cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), color1, 6)
		cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), color2, 2)
		cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), color1, 6)
		cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), color2, 2)
		cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), color1, 6)
		cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), color2, 2)

		# Palm
		cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), color1, 6)
		cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), color2, 2)
		cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), color1, 6)
		cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), color2, 2)
		cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), color1, 6)
		cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), color2, 2)
		cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), color1, 6)
		cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), color2, 2)
		cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), color1, 6)
		cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), color2, 2)
		cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), color1, 6)
		cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), color2, 2)
		cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), color1, 6)
		cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), color2, 2)

	# Key Points
	for index, landmark in enumerate(landmark_point):
		if index == 0:  # 手首1
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 1:  # 手首2
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 2:  # 親指：付け根
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 3:  # 親指：第1関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 4:  # 親指：指先
			cv.circle(image, (landmark[0], landmark[1]), 8, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 8, color1, 1)
		if index == 5:  # 人差指：付け根
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 6:  # 人差指：第2関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 7:  # 人差指：第1関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 8:  # 人差指：指先
			cv.circle(image, (landmark[0], landmark[1]), 8, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 8, color1, 1)
		if index == 9:  # 中指：付け根
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 10:  # 中指：第2関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 11:  # 中指：第1関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 12:  # 中指：指先
			cv.circle(image, (landmark[0], landmark[1]), 8, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 8, color1, 1)
		if index == 13:  # 薬指：付け根
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 14:  # 薬指：第2関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 15:  # 薬指：第1関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 16:  # 薬指：指先
			cv.circle(image, (landmark[0], landmark[1]), 8, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 8, color1, 1)
		if index == 17:  # 小指：付け根
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 18:  # 小指：第2関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 19:  # 小指：第1関節
			cv.circle(image, (landmark[0], landmark[1]), 5, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 5, color1, 1)
		if index == 20:  # 小指：指先
			cv.circle(image, (landmark[0], landmark[1]), 8, color2, -1)
			cv.circle(image, (landmark[0], landmark[1]), 8, color1, 1)
	return image


def rectangle_overlap(brects, brect):
	max_overlap = 0
	label = brect[1]
	brect = brect[0]
	for b in brects:
		tmp_label = b[1]
		b = b[0]
		dx = min(b[2], brect[2]) - max(b[0], brect[0])
		dy = min(b[3], brect[3]) - max(b[1], brect[1])
		if dx >= 0 and dy >= 0:
			overlap = dx * dy / ((brect[2] - brect[0]) * (brect[3] - brect[1]))
			if tmp_label != label:
				overlap -= 0.2 # reduce overlap value if hand is not the same (left | right)
			if overlap > max_overlap:
				max_overlap = overlap
	return max_overlap


class CvHandTracking(object):
	"""docstring for CvHandTracking"""
	def __init__(self,
		max_num_hands=2,
		min_detection_confidence=0.6,
		min_tracking_confidence=0.2,
		history_length=16,
		color1=(3, 3, 3),
		color2=(252, 252, 252),
		img_flip=True,
		show_fps=True):
		self.mp_hands = mp.solutions.hands
		self.mp_drawing = mp.solutions.drawing_utils
		self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
			min_tracking_confidence=min_tracking_confidence)
		self.cvFpsCalc = CvFpsCalc(buffer_len=10)
		self.keypoint_classifier = KeyPointClassifier()
		with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
			self.keypoint_classifier_labels = csv.reader(f)
			self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]
		self.point_history = deque(maxlen=history_length)
		self.color1 = color1
		self.color2 = color2
		self.img_flip = img_flip
		self.show_fps = show_fps

		self.hand_sign_prio = [1, 0, 4, 2, 3] # hand_sign_id priority


	def process(self, img, color1=None, color2=None, img_flip=None, show_fps=None):
		if color1 is not None:
			self.color1 = color1
		if color2 is not None:
			self.color2 = color2
		if img_flip is not None:
			self.img_flip = img_flip
		if show_fps is not None:
			self.show_fps = show_fps

		if self.img_flip:
			img = cv.flip(img, 1)
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img.flags.writeable = False
		results = self.hands.process(img)
		img.flags.writeable = True
		img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
		if results.multi_hand_landmarks:
			data = {"landmark_list": [], "label": [], "brect": [], "hand_sign_id": [], "prio": [], "area": []}
			for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
				brect = calc_bounding_rect(img, hand_landmarks)
				landmark_list = calc_landmark_list(img, hand_landmarks)
				pre_processed_landmark_list = pre_process_landmark(landmark_list)
				# Hand sign classification
				hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

				data["landmark_list"].append(landmark_list)
				data["label"].append(handedness.classification[0].label[0:])
				data["brect"].append(brect)
				data["hand_sign_id"].append(hand_sign_id)
				data["prio"].append(self.hand_sign_prio[hand_sign_id])
				data["area"].append((brect[2] - brect[0]) * (brect[3] - brect[1]))
			df = pd.DataFrame(data=data)
			df = df.sort_values(by=["prio", "area"], ascending=False).reset_index(drop=True)

			brects = []
			for i in df.index:
				landmark_list = df["landmark_list"][i]
				label = df["label"][i]
				brect = df["brect"][i]
				hand_sign_id = df["hand_sign_id"][i]
				
				overlap = rectangle_overlap(brects, (brect, label))
				if overlap > 0.7:
					# print("overlap", overlap)
					continue

				if hand_sign_id == 2:  # Point gesture
					self.point_history.append(landmark_list[8])
				else:
					self.point_history.append([0, 0])
				brects.append((brect, label, hand_sign_id))

				# Drawing part
				cv.rectangle(img, (brect[0], brect[1]), (brect[2], brect[3]), self.color1, 1)
				img = draw_landmarks(img, landmark_list, color1=self.color1, color2=self.color2)
				img = draw_info_text(img, brect, label, self.keypoint_classifier_labels[hand_sign_id], color1=self.color1, color2=self.color2)

			# if len(brects) > 1:
			# 	nb_peace = len([b for b in brects if b[2] == 4]) # Peace gesture
			# 	print("nb_peace", nb_peace)
			# 	if nb_peace >= 2:
			# 		print("Playing sound")
			# 		playsound('blbl.mp3')
		else:
			self.point_history.append([0, 0])

		img = draw_point_history(img, self.point_history)
		fps = self.cvFpsCalc.get()
		if self.show_fps:
			cv.putText(img, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, self.color1, 4, cv.LINE_AA)
			cv.putText(img, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, self.color2, 2, cv.LINE_AA)
		return img


if __name__ == '__main__':
	cap_width = 960
	cap_height = 540

	# Opencv setup
	cap = cv.VideoCapture(0)
	cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

	hand_tracking = CvHandTracking()

	while True:
		key = cv.waitKey(10)
		if key == 27:  # ESC
			break
		ret, img = cap.read()
		if not ret:
			break
		img = hand_tracking.process(img)
		cv.imshow('Hand tracking', img)
	cap.release()
	cv.destroyAllWindows()
