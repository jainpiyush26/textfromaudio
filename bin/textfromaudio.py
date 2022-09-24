#!/usr/bin/env python
# stdlib imports
from argparse import ArgumentParser
import os
import pprint
import json
import datetime

# external imports
import whisper
from whisper.tokenizer import LANGUAGES

def args_parsing():
    """ Arguments
    """
    args = ArgumentParser("Convert audio files to text")
    args.add_argument("--audio", "-a", required=True,
                      help="Location of the audio file")
    args.add_argument("--output", "-o", 
                      help="Location where the text files should" \
                           " be exported")
    args.add_argument("--dryrun", "-d", help="Don't do anything, just print")

    return args.parse_args()


def main():
    """
    Main Function
    """
    args = args_parsing()
    audio_path = args.audio
    export_path = args.output
    dryrun = args.dryrun

    if dryrun:
        pprint.pprint("INFO: We are looking at {0}".format(audio_path))
        if export_path:
            pprint.pprint("INFO: Exporting output to {1}".format(export_path))
        return

    if not os.path.exists(audio_path):
        raise RuntimeError("The audio path provided does not exist!")

    start_time = datetime.datetime.now()

    model = whisper.load_model("base")

    result = model.transcribe(audio_path, fp16=False)

    end_time = datetime.datetime.now()

    time_taken = datetime.timedelta(seconds=(end_time-start_time).seconds)
    lang = LANGUAGES.get(result.get("language"))
    output_dict = {"language":lang,
                   "text":result.get("text"),
                   "processing time":str(time_taken)}

    if export_path:
        with open(export_path, "w") as file_write:
            json.dump(output_dict, file_write, indent=4)
    else:
        pprint.pprint(output_dict)

if __name__ == "__main__":
    main()
