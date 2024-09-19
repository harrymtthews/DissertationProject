import numpy as np
from skimage.draw import line
from skimage.morphology import thin

import matplotlib.pyplot as plt
from matplotlib import transforms
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

import xml.etree.ElementTree as ET
from io import StringIO

import cv2
import json

from itertools import chain
import random
import math

import glob, os


# Largely derrived from 'parse_inkml', credit to [6] in report references

inkml_file_abs_path = 'datasets/IAMonDo-db-1.0/001.inkml'


def get_traces_data(inkml_file_abs_path):

    # Parses XML data, converts traces to global image cooords, finds the traces relating to markups in XML.

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()

    author_details = {}
    for ann in root.findall('annotation'):
        if ann.get('type').startswith('author'):
            author_details[ann.get('type')] = ann.text

    matrix = root.find('definitions').find('canvasTransform').find('mapping').find('matrix')
    if matrix is not None:
        matrix = matrix.text.split(',')
        matrix = np.array([ float(x) for d in matrix[:2] for x in d.split(' ')[:2] ])

    # Stores traces_all with their corresponding id
    traces_all = {} 
    for trace_tag in root.findall('trace'):
        id_ = trace_tag.get('{http://www.w3.org/XML/1998/namespace}id')
        coords = (trace_tag.text).replace('\n', '').split(',')

        x,y = [float(x) for x in coords[0].split(' ') if x][:2]
        
        traces_all[id_] = [[x,y]] 
        
        try:
            vx, vy = [ float(x) for x in coords[1].split("'") if len(x) > 0 ][:2]
            px,py = traces_all[id_][-1] 
            traces_all[id_].append([px + vx, py + vy])
        except Exception as e:
            continue

        try:
            ax, ay = [ float(x) for x in coords[2].split('"') if len(x) > 0 ][:2]
            vx+=ax
            vy+=ay
            traces_all[id_].append([px + vx, py + vy])
        except Exception as e:
            continue

        for coord in coords[3:]:
            rel_coords =  [float(n) for n in coord.replace('-', ' -').split(" ") if n]
            vx += rel_coords[0]
            vy += rel_coords[1]

            px,py = traces_all[id_][-1] 
            traces_all[id_].append([px + vx, py + vy])

    markups = root.find('traceView')[-1]
    markups = markups if markups[0].text == 'Marking' else False

    return traces_all, markups, matrix


def transform_points(traces, M):

    # Rescales all of the traces to pixels within an image
    
    min_x, min_y = np.inf, np.inf
    max_y, max_x = -np.inf, -np.inf 
    
    if M is not None:
        M = M.reshape((2,2))
    else:
        M = np.array([[1,0],[0,1]])

    for t_id in traces:
        pts = traces[t_id]
        pts = np.array([list(np.dot(M, x)) for x in pts] , dtype=np.int32)
        y,x = zip(*pts)
        min_x, min_y = min(min(x), min_x), min(min(y), min_y)
        max_x, max_y = max(max(x), max_x), max(max(y), max_y)
        traces[t_id] = pts
        
    width   = max_y if min_y > 0 else max_y - min_y
    height  = max_x if min_x > 0 else max_x - min_x
  
    offset = 25
    width = width + 2*offset
    height = height + 2*offset
   
    offset_x, offset_y = min_x, min_y

    for t_id in traces:
        pts = traces[t_id]
        pts = np.array([(y - offset_y + offset if offset_y < 0 else y + offset, 
                            x - offset_x + offset if offset_x < 0 else x + offset) 
                        for (y,x) in pts], dtype=np.int32)
        traces[t_id] = pts.reshape((-1,1,2))

    return traces, height, width


def extract_markings(traces, markings_tree, markings, size, file_id, pad):

    # Takes the global traces list and the markings subtree, writes a set of images representing the lot
    
    # Start by creating an empty image
    page_marking = np.zeros(size, np.uint8)
    page_other = np.zeros(size, np.uint8)
    page_incl_traces = set()

    mark_opening = ('[', 'top-left')
    mark_closing = (']', 'bottom-right')

    for i, mark in enumerate(markings):
        img_marking = np.zeros(size, np.uint8)
        img_other = np.zeros(size, np.uint8)
        mark_objs = []
        mark_bbs = []
        
        two_part = []
        mark_count = 0

        for traceview in markings_tree:
            if traceview.tag == 'traceView' and traceview[0].text == mark:
                mark_traces = []
                
                # If the marking is an opening, add it to the two_part list
                if traceview[1].text in mark_opening:
                    two_part.append(mark_count)

                # Add all traces of the marking to a list
                for x in traceview:
                    if 'traceDataRef' in x.attrib:
                        mark_traces.append(x.attrib['traceDataRef'][1:])
                
                # If the marking is a closing and there is already an opening, combine them
                if two_part and traceview[1].text in mark_closing:
                    mark_objs[two_part.pop()] += mark_traces
                else:
                    mark_objs.append(mark_traces)
                    mark_count += 1
                    
        # Find a bounding box for each object
        ts = np.empty((0, 0))
        for obj in mark_objs:
            ts = np.concatenate([traces[t_id] for t_id in obj], axis=0)
            mark_bbs.append(cv2.boundingRect(ts))

        # Draw them on the segmented image diff.
        # for bb in mark_bbs:
        #     cv2.rectangle(img_segmented, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), 255, -1)

        # Determing which traces are a part of the markup and which aren't
        incl_traces = set(chain.from_iterable(mark_objs))
        excl_traces = set(traces.keys()) - incl_traces

        page_incl_traces.update(incl_traces)
        
        # Draw each set of traces
        for t_id in incl_traces:
            cv2.polylines(img_marking, [traces[t_id]], False, 255, thickness=1, lineType=cv2.LINE_AA)

        for t_id in excl_traces:
            cv2.polylines(img_other, [traces[t_id]], False, 255, thickness=1, lineType=cv2.LINE_AA)
        
        # For each marking in the list
        for j, bb in enumerate(mark_bbs):

            # Create two new empty images with the same size as the bounding box + padding in each direction
            h, w = bb[3]+2*pad, bb[2]+2*pad
            img_marking_out = np.zeros((h, w), np.uint8)
            img_other_out = np.zeros((h, w), np.uint8)

            lower_x = max(0, bb[0] - padding)
            upper_x = min(size[1], bb[0]+bb[2]+padding)
            offset_x = padding - bb[0] if lower_x == 0 else 0

            lower_y = max(0, bb[1] - padding)
            upper_y = min(size[0], bb[1]+bb[3]+padding)
            offset_y = padding - bb[1] if lower_y == 0 else 0

            img_marking_out[offset_y:offset_y+(upper_y-lower_y), offset_x:offset_x+(upper_x-lower_x)] = img_marking[lower_y:upper_y, lower_x:upper_x]
            img_other_out[offset_y:offset_y+(upper_y-lower_y), offset_x:offset_x+(upper_x-lower_x)] = img_other[lower_y:upper_y, lower_x:upper_x]

            cv2.imwrite(f'{output_dir}/{markings[i]}/Marking/{file_id}-{j}.png', img_marking_out)
            cv2.imwrite(f'{output_dir}/{markings[i]}/Other/{file_id}-{j}.png', img_other_out)
    
    page_excl_traces = set(traces.keys()) - page_incl_traces

    for t_id in page_incl_traces:
        cv2.polylines(page_marking, [traces[t_id]], False, 255, thickness=1, lineType=cv2.LINE_AA)

    for t_id in page_excl_traces:
        cv2.polylines(page_other, [traces[t_id]], False, 255, thickness=1, lineType=cv2.LINE_AA)

    
    cv2.imwrite(os.path.join(output_dir, 'Page', 'Marking', f'{file_id}.png'), page_marking)
    cv2.imwrite(os.path.join(output_dir, 'Page', 'Other', f'{file_id}.png'), page_other)

    return


def extract_other(traces, markings_tree, markings, size, file_id):

    img_other = np.zeros(size, np.uint8)

    return


def run_test():
    all_traces, markups, m = get_traces_data('datasets/IAMonDo-db-1.0/001.inkml')
    all_traces, height, width = transform_points(all_traces, m)

    if markups:
        extract_markings(all_traces, markups, ['Marking_Underline'], (height, width))


if __name__ == '__main__':

    padding = 50

    markup_types = ['Marking_Underline', 'Marking_Encircling', 'Marking_Connection', 'Marking_Sideline', 'Marking_Bracket', 'Marking_Angle']

    input_dir = 'datasets/IAMonDo-db-1.0/'
    output_dir = 'datasets/IAMonDo-Processed/'
    os.makedirs(output_dir, exist_ok=True)

    # Set up output directories
    # For each markup_type, create folders for marked and unmarked folders

    for markup in markup_types+['Page']:
        markup_dir = os.path.join(output_dir, markup)
        os.makedirs(markup_dir, exist_ok=True)
        os.makedirs(os.path.join(markup_dir, 'Marking'), exist_ok=True)
        os.makedirs(os.path.join(markup_dir, 'Other'), exist_ok=True)

    for idx, input_path in enumerate(sorted(glob.glob(os.path.join(input_dir, "*.inkml")))):
        
        file_id = input_path.split('\\')[-1][:-6]
        
        print(file_id)

        all_traces, markups, m = get_traces_data(input_path)
        all_traces, height, width = transform_points(all_traces, m)

        if markups:
            extract_markings(all_traces, markups, markup_types, (height, width), file_id, padding)
