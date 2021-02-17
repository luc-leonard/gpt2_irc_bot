import argparse
import textwrap

import irc
import irc.bot
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
import random


class IrcBot(irc.bot.SingleServerIRCBot):

    def __init__(self, channel: str, nickname: str, server: str, generator, wakeword,
                 port=6667):
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port)], nickname, nickname)

        self.channel = channel
        self._generator = generator
        self._wakeword = wakeword

    def on_nicknameinuse(self, c: irc.client, e):
        c.nick(c.get_nickname() + "_")

    def on_welcome(self, c, e):
        c.join(self.channel)

    def on_privmsg(self, c: irc.client.ServerConnection, e: irc.client.Event):
        ...

    def on_pubmsg(self, c, e):
        text = e.arguments[0]
        if len(text) > 100:
            text = text[:99]
        args = text.split()
        if args[0] == '!' + self._wakeword:
            text = self._generator(' '.join(args[1:]),
                                   max_length=100,
                                   device=0,
                                   do_sample=False,
                                   top_p=random.randint(50, 100) / 100,
                                   temperature=1,
                                   top_k=500,
                                   num_beams=4,
                                   no_repeat_ngram_size=2,
                                   num_return_sequences=1)[0]['generated_text']
            wrapped = textwrap.fill(text, 400)
            for line in wrapped.splitlines(keepends=False):
                c.privmsg(e.target, line)


    def on_dccmsg(self, c, e):
        pass

    def on_dccchat(self, c, e):
        pass

    def do_command(self, e, cmd):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", help="the model")
    parser.add_argument("--name", help="the name")
    parser.add_argument("--wakeword",
                        help="wakeword")
    
    parser.add_argument("--server", help="the model")
    parser.add_argument("--channel", help="the model")

    return parser.parse_args()

def main():
    args = parse_arguments()
    print("loading generators...")
    model = GPT2LMHeadModel.from_pretrained(args.model).to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained("antoiloui/belgpt2")
    print('generator loaded...')
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    bot = IrcBot(args.channel, args.name, args.server,
                 generator, args.wakeword)

    bot.start()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
