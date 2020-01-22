import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # From January 21, 2020 Digital Daily:
  '지난달 국회 과학기술정보방송통신위원회는 정보통신방송법안심사소위를 열고 댓글과 실시간급상승검색어 조작을 막는 정보통신망법 개정안에 합의한 바 있다.',
  '모든 매크로를 악성프로그램으로 볼 수 있나. 현재 기술적 수준으로 어디까지 막을 수 있나.',
  '부당한 목적으로 여론형성 집단이 관철하는 콘텐츠가 무엇인지 생각해봐야 한다.',
  # From January 21, 2020 NEWSIS:
  '국내 연구진이 살균이나 소독에 흔하게 쓰는 염소를 기존보다 저렴하고 효율적으로 만드는 방법을 개발했다.',
  '또 바닷물처럼 염소 이온을 많이 포함하거나 반대로 염소 이온 농도가 낮아도 높은 효율을 보였다.',
  # From January 21, 2020 매일경제:
  '날계란을 집거나 피아노를 치고 가위질을 하는 등 일상생활의 다양한 손동작을 자유자재로 구사할 수 있는 로봇 손이 개발됐다.',
  '특히 손가락을 움직이는 구동부를 손바닥 내부에 장착해 모듈화 했다.',
  '연구진은 이번 연구 결과를 토대로 지난해 국제특허를 출원했다.',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
  main()
