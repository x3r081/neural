"""Terminal chat for the separate Qwen server, retaining reasoning history."""
import argparse
import json
import os
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8001/v1')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--no-thinking', action='store_true')
    args = ap.parse_args()
    history = []
    print('Neural Qwen chat. /reset clears history; /quit exits.')
    while True:
        try:
            prompt = input('\nyou> ').strip()
        except (KeyboardInterrupt, EOFError):
            return
        if prompt in {'/quit', '/exit'}:
            return
        if prompt == '/reset':
            history.clear()
            continue
        if not prompt:
            continue
        messages = history + [{'role': 'user', 'content': prompt}]
        body = {'model': 'qwen3.6-35b-a3b-neural', 'messages': messages, 'stream': True,
                'max_tokens': args.max_tokens, 'chat_template_kwargs': {
                    'enable_thinking': not args.no_thinking, 'preserve_thinking': True}}
        req = urllib.request.Request(args.url.rstrip('/')+'/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
            'Authorization': 'Bearer '+os.environ.get('NEURAL_API_KEY', 'local')})
        content, reasoning, last_mode = '', '', None
        try:
            with urllib.request.urlopen(req, timeout=7200) as response:
                for line in response:
                    line = line.decode('utf-8').strip()
                    if not line.startswith('data: ') or line == 'data: [DONE]':
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get('error'):
                        raise RuntimeError(chunk['error']['message'])
                    for choice in chunk.get('choices', []):
                        delta = choice.get('delta', {})
                        for field, label in [('reasoning_content', 'thinking'), ('content', 'answer')]:
                            text = delta.get(field, '')
                            if not text:
                                continue
                            if last_mode != label:
                                print('\n'+label+'> ', end='', flush=True)
                                last_mode = label
                            print(text, end='', flush=True)
                            if field == 'content':
                                content += text
                            else:
                                reasoning += text
                    if chunk.get('usage'):
                        usage = chunk['usage']
                        print(f"\n[{usage['completion_tokens']} generated tokens]")
            history = messages + [{'role': 'assistant', 'content': content,
                                   'reasoning_content': reasoning}]
        except (OSError, RuntimeError) as exc:
            print(f'\nRequest failed: {exc}')
        except KeyboardInterrupt:
            print('\nCancelled; this turn was not added to history.')


if __name__ == '__main__':
    main()
