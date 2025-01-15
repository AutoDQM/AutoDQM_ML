import csv
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Read and print the first row of a CSV file.')
    parser.add_argument('csvfile', type=str, help='Path to the CSV file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the CSV file and print the first row
    with open(args.csvfile, newline='') as csvfile:
        reader = csv.reader(csvfile)
        first_row = next(reader)
        print(first_row)

if __name__ == '__main__':
    main()
