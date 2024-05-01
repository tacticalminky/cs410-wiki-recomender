# cs410-wiki-recomender

School project for CS 410

## Requirements and Dependencies

Please install the dependencies listed in the `requirements.txt` file. 

The data collection component is relient on the UNIX `/tmp` directory for temporary file during the web crawling portion. If you want to change the location, you can do so in the `src/crawlerWorker.py` file in the `OUTPUT_DIR` variable.

## How to Run

There are sample documents stored in the `./data/sample` directory. If you want to use those, please copy/move them to the `./data` directory.

To run the data collection component, run `python src/build.py`

To run the query component with user input, run `python src/run.py`

To run the test queries, run `python src/test_run.py`
