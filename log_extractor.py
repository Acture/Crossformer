import argparse
import collections
import io
import os
import re

import pandas as pd


def read_command_line_args():
	parser = argparse.ArgumentParser(description="Python Argparse example")
	parser.add_argument("--path", default="." + os.sep + "logs", help="relative path to the work directory", type=str)
	parser.add_argument("--prefix", default="exp_", help="prefix for the log files", type=str)
	parser.add_argument("pattern", default=".*", help="Pattern to analyze", type=str)
	parser.add_argument("-ext", "--extension", default="log", help="file extension to analyze", type=str)
	parser.add_argument('-s', '--save', type=str, default='', help='Save the result. Provide the save destination')
	parser.add_argument("-c", "--copy", action='store_true', help="save the result to the clipboard")
	parser.add_argument("-ni", "--no_index", action='store_true', help="save the result with the index")
	parser.add_argument("-nh", "--no_header", action='store_true', help="save the result with the header")

	args = parser.parse_args()

	return args


def get_last_line_from_files(file_info):
	last_lines_data = {}
	for file_path, name in file_info:
		try:
			with open(file_path, 'rb') as file:
				file.seek(-2, os.SEEK_END)
				while file.read(1) != b'\n':
					file.seek(-2, io.SEEK_CUR)
				last_line = file.readline().decode()
				last_lines_data[name] = extract_mse_mae(last_line)
		except IOError:
			print(f"Error opening file: {file_path}")

	return last_lines_data


def extract_mse_mae(last_line):
	error_data = collections.OrderedDict()
	if last_line:
		mse_error, mae_error = [float(metric.split(':')[1]) for metric in last_line.split(', ')]
		error_data['mse'] = mse_error
		error_data['mae'] = mae_error

	return error_data


def analyze_log_files():
	args = read_command_line_args()
	path = args.path
	prefix = args.prefix
	pattern = args.pattern
	extension = args.extension
	save = args.save
	copy = args.copy
	index = not args.no_index
	header = not args.no_header

	regex_pattern = prefix + "(" + pattern + ")\." + extension + "$"

	print_pattern_and_options(regex_pattern, save, copy, index, header)

	all_files = os.listdir(path)
	file_info = [(path + os.sep + file, re.match(regex_pattern, file).group(1)) for file in all_files if
	             os.path.isfile(path + os.sep + file) and re.match(regex_pattern, file)]
	last_lines_data = get_last_line_from_files(file_info)
	df_flat = construct_dataframe(last_lines_data)
	reshaped_dataframe = reshape_dataframe(df_flat)
	print(reshaped_dataframe)
	if save:
		df_flat.to_csv(save, index=index, header=header)

	if copy:
		reshaped_dataframe.to_clipboard(excel=True, index=index, header=header)


def print_pattern_and_options(log_pattern, save, copy, index, header):
	print(f"Match: {log_pattern}", end="")
	if save:
		print(f", Save: {save}", end=", ")
	if copy:
		print(f", Copy: {copy}", end=", ")
	if save or copy:
		print(f"Index: {index}, Header: {header}")
	else:
		print("")


def construct_dataframe(last_lines_data):
	df = pd.DataFrame.from_dict(last_lines_data, orient='index')
	df.index = pd.MultiIndex.from_tuples(df.index.str.split('_').tolist(),
	                                     names=('dataset', 'predict length', 'input length'))
	df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)
	df.index = df.index.set_levels(df.index.levels[2].astype(int), level=2)
	df.reset_index(inplace=True)
	return df


def reshape_dataframe(df_flat):
	reshaped_dataframe = df_flat.pivot_table(index=['dataset', 'predict length'], columns='input length')
	reshaped_dataframe.columns = reshaped_dataframe.columns.swaplevel(0, 1)
	reshaped_dataframe.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
	return reshaped_dataframe.reindex(columns=['mse', 'mae'], level=1)


if __name__ == "__main__":
	analyze_log_files()
