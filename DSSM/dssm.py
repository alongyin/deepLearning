import argparse
from config import Config






if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",default="bert",type=str,help="train/predict")
    ap.add_argument("--mode",default="train",type=str,help="train/predict")
    ap.add_argument("--file",default="./results/input/test",type=str,help="train/predict")
