# 읽기 전용 진동 특성 탐색 분석

## 목적과 검증 범위

900·1200·1800RPM의 실제 진동 기록을 **RPM × 날짜 × 조건** 단위로 살펴보는 EDA 도구다. AE나 비교 모델을 호출하지 않는다. 결함 후보 자동 분류, 규칙 생성, 웹 연결, 재학습, 임계값 변경은 포함하지 않는다.

이번 Codex 환경에는 프로젝트의 실제 MAT 원본이 없다(설치 패키지의 MAT 테스트 파일은 프로젝트 데이터가 아니다). 실제 정상/결함의 특징 차이, 엔벨로프 필터의 현장 적합성, 결함 판별 성능은 확인하지 못했다. 아래 테스트는 합성 신호의 수치 계산과 파일 처리·출력 검증이다. **실제 데이터 분석 완료나 결함 진단 성공을 의미하지 않는다.**

## WSL 실행 전 확인

1. 명령을 실행할 곳이 `.git`이 있는 실제 소스 저장소인지 `git rev-parse --show-toplevel`로 확인한다. `~/capstone`이 데이터 폴더인 경우 소스 폴더와 구분한다.
2. 기존 웹 전용 환경 `~/capstone-web-venv/bin/python`에서 NumPy·SciPy·Matplotlib을 import할 수 있어야 한다. 이 도구는 모델 가중치나 번들을 요구하지 않으며 패키지를 자동 설치하지 않는다.
3. 원본 **데이터 폴더**를 `--data-root`에 지정한다. 가상환경·라이브러리·전체 홈 폴더가 아니라 날짜별 측정 데이터가 있는 좁은 폴더를 사용한다. 여러 개의 데이터 루트를 함께 지정할 수 있다.
4. 지원되는 원본 구조는 `…/20260912/20260912_900RPM/IR/Vib/20260912_144039_Vib.mat` 형태다. 날짜/RPM 폴더, 정확한 조건 폴더, `Vib`, 날짜·시각 파일명을 확인한다. 다른 구조는 임의 추정하지 않고 제외한다.
5. 기본적으로 모든 대상 RPM·조건·날짜를 탐색한다. 실제 보유 데이터에 빠진 조합은 JSON의 `missing_strata_by_observed_date`에 표시한다. 같은 이름이어도 날짜나 RPM이 맞지 않으면 제외된다.
6. 출력은 Git 저장소 및 데이터 폴더와 분리된 새 폴더로 지정한다. 기본값은 `~/fault-signatures/<UTC 타임스탬프>/`다. 기존 폴더에는 쓰지 않는다. CSV와 PNG가 많으므로 여유 공간을 확인한다.
7. 센서 단위, 감도, 방향, 측정 부착 조건 및 실제 RPM의 신뢰도를 별도로 확인한다. 이 도구는 단위를 가속도나 속도로 추측하거나 변환하지 않는다.

## 실행 명령

먼저 소스 저장소에서 새 브랜치를 가져온다. 아래 `cd`와 데이터 경로는 실제 WSL 경로로 바꾼다.

```bash
cd /실제/Git/저장소
git fetch origin
git switch --track origin/feat/fault-signature-eda
```

브랜치가 이미 로컬에 있으면 `git switch feat/fault-signature-eda`를 사용한다. 로컬 변경이 있으면 강제 전환·초기화를 하지 않는다.

실제 분석은 한 명령으로 실행한다.

```bash
"$HOME/capstone-web-venv/bin/python" -B -m scripts.analyze_fault_signatures \
  --data-root /실제/원본/MAT/데이터폴더 \
  --files-per-stratum 3 \
  --output "$HOME/fault-signatures/$(date -u +%Y%m%dT%H%M%S)-eda"
```

예를 들어 실제 폴더가 `~/capstone/data/20260912`, `20260916`, `20260919`라면 `--data-root "$HOME/capstone/data"`로 지정한다. 데이터가 다른 폴더에 있으면 그 경로를 사용한다. `--output`을 생략하면 충돌 가능성이 낮은 마이크로초 단위 새 경로를 자동 생성한다.

선택 사항: `--envelope-band 2000 8000`, `--order-tolerance-hz 1.0`. 필터 대역 변경은 별도 출력 폴더에서 민감도를 탐색하는 용도다. 결과가 잘 분리되는 대역을 사후 골라 검증된 성능으로 제시하면 안 된다.

## 입력 검증과 샘플 선정

- 로더는 기존 `vibration.web_inference.preview_mat`를 그대로 재사용한다. 모델이나 DB는 열지 않는다. 이 로더의 기존 형식인 `signals=(6,65536)`, 단일 `fs=25600Hz`, 20MiB 이하를 요구한다.
- 전체 6채널의 실수·유한값 및 극단값 한계(절댓값 1e100)를 검증하고 CH3(인덱스 2)만 분석한다. 다른 채널, 측정 길이, 샘플링 주파수의 파일을 자동 보정하지 않는다. 로더가 거부한 파일의 MAT 변수명·shape·class는 읽을 수 있는 경우 JSON에 남는다.
- 유효 파일의 실제 샘플 수·fs·측정 길이·채널 수를 JSON과 특징량 CSV에 기록한다. 현재 지원 형식의 측정 길이는 65536/25600=2.56초다. 채널의 물리적 방향·단위는 확인하지 않는다.
- 정확한 대상 조건: `Normal1`, `Normal2`, `IR`, `OR`, `Roller`, `Looseness0.05`, `Looseness0.1`, `LoosenessFull`, `Unbalance_-1`, `Unbalance_-2`. `Normal3`, `Normal`, 부분 일치 문자열, 다른 센서 폴더는 포함하지 않는다.
- 각 **RPM·날짜·정확한 조건**의 정렬된 파일 경로에서 `linspace(0,n-1,min(k,n))`의 정수 인덱스를 선택한다. 기본 k=3으로 앞·중간·뒤 파일을 선택한다. k=1이면 첫 파일이다. 일정한 파일명 규칙에서 이는 측정 시간순이다.
- 오류 파일을 다른 파일로 조용히 교체하지 않는다. `not_selected`(샘플 수 제한)와 `excluded`(조건 불명 또는 읽기·품질·계산 오류)를 구분한다. 파일이 k개보다 적으면 있는 파일만 선택한다.
- 원본 SHA256을 읽기 전후 비교하며 외부에서 바뀐 파일은 제외한다. 원본 쓰기·이름 변경·수정은 수행하지 않는다.
- 모든 탐색 파일과 상태·사유·선정 경로를 `analysis.json` 및 `selection_and_exclusions.csv`에 기록한다. 누락된 전체 날짜는 원본 폴더에 없으면 알 수 없으며, JSON의 누락 조합은 관찰된 날짜를 기준으로 한다.

## 계산 정의

### 시간 특징

`rms_raw = sqrt(mean(x²))`, `peak_raw = max(abs(x))`, `mean_raw = mean(x)`를 기록한다. 비교 그래프의 RMS·Peak·Crest Factor는 DC를 제거한 `y=x-mean(x)` 기준이다.

- `rms_ac = sqrt(mean(y²))`
- `peak_ac = max(abs(y))`
- `crest_factor_ac = peak_ac/rms_ac`

RMS가 0이면 Crest Factor는 결측으로 남기며 무한대나 임의의 0으로 바꾸지 않는다. 센서가 교정됐다는 증거가 없어 단위는 raw units다. 큰 DC 차이는 raw/ac RMS 및 mean을 함께 확인한다.

### FFT와 1X·2X

AE의 8구간 스펙트럼과 달리 EDA는 **전체 65536샘플**을 사용한다. 평균 제거 → 대칭 `np.hanning(N)` → rFFT → 절댓값/윈도 합 → DC·Nyquist를 제외한 양의 주파수 2배로 진폭을 계산한다. FFT 값은 피크 진폭이며 PSD와 구분한다. 전체 기록 기준 Δf=fs/N=**0.390625Hz**다. FFT 저장·비교 범위는 0–10000Hz다.

1X=RPM/60, 2X=2RPM/60으로 계산한다. 900RPM은 15/30Hz, 1200RPM은 20/40Hz, 1800RPM은 30/60Hz다. 기본 허용 범위는 각각 **±1.0Hz**다. 해당 FFT bin 중심 중 최대 진폭과 그 bin의 주파수를 저장한다. 보간·order tracking은 하지 않는다. 명목 범위 밖의 피크를 끌어오지 않는다. bin 양자화를 포함한 유효 반폭은 `tolerance + Δf/2`로 별도 기록한다. 범위 설정은 최소 1bin 이상이며 1X·2X 탐색 구간이 겹치지 않아야 한다.

이는 폴더의 **명목 RPM**을 이용한 측정이며 타코미터 기반 실제 회전수 추적이 아니다. 탐색 구간 내 다른 주파수 성분을 1X나 2X로 오인할 수 있으므로 실제 RPM·스펙트럼을 함께 확인해야 한다.

주요 피크는 양의 주파수 국소 최대점 중 진폭 상위 10개다. prominence 임계값이나 결함 진단 의미를 부여하지 않는다. DC·경계점은 국소 피크 목록에 포함하지 않는다.

### 대역별 에너지

| 이름 | 주파수 대역 |
| --- | --- |
| low | 0 ≤ f < 200Hz |
| mid | 200 ≤ f < 2000Hz |
| high | 2000 ≤ f ≤ 10000Hz |

평균 제거 후 같은 Hann 윈도를 사용한 단측 `scipy.signal.periodogram(..., scaling='density', detrend=False)`를 계산한다. `energy_<band> = sum(PSD[band]) × Δf`다. 값은 **대역 평균제곱 진동 크기(raw units²)**로, 시간적분 에너지나 FFT 진폭 합이 아니다. Hann 윈도에 의한 경계 부근 누설은 남는다. 대역 경계는 탐색용 고정 설정이며 특정 결함과 자동 연결하지 않는다.

### 엔벨로프

정상 대조군을 포함한 모든 조건에 같은 방법을 적용한다. 기본 carrier 대역은 **2000–8000Hz**다. `0 < low < high < fs/2(12800Hz)`를 검사하고, 4차 Butterworth bandpass를 SOS로 만든 뒤 `sosfiltfilt`로 처리한다. 이후 전체 필터 신호의 Hilbert 절댓값을 구하고 양끝 **0.1초씩** 버린 뒤 평균 제거·Hann FFT를 수행한다. 잘라낸 후의 실제 Δf도 파일별 기록한다. 엔벨로프 저장·표시 범위는 0–1000Hz다.

이 대역이 수치적으로 적용 가능하다는 것과 베어링 공진을 잘 포착한다는 것은 다르다. `envelope_filter_response.csv`에는 왕복 필터의 진폭 응답 `|H|²`를 저장한다. `carrier_power_fraction`에는 원신호 PSD에서 해당 carrier 대역이 차지하는 비율을 기록한다. 필터 대역·주변 에너지, 센서 대역폭, 포화·노이즈, 날짜별 재현성 및 다른 carrier 대역에서의 민감도를 실제 데이터로 확인해야 한다. 에너지가 거의 없는 대역에서 얻은 엔벨로프를 유효한 근거로 삼지 않는다. 기본 대역을 실제로 타당성이 검증된 대역이라고 표시하지 않는다.

베어링 형상·전동체 수·접촉각·피치 지름이 없어 **BPFO/BPFI/BSF/FTF를 계산하거나 IR/OR/Roller의 특정 결함 주파수로 단정하지 않는다.** 양끝 제거도 모든 필터·Hilbert 경계 영향을 완전히 제거한다는 보장은 아니다.

## 결과 파일

```text
<새 출력 폴더>/
  analysis.json                     # 설정·버전·선정 목록·상태·제외 사유·해시·제한
  selection_and_exclusions.csv       # 전체 탐색/샘플 제한/제외 목록
  file_features.csv                  # 분석 성공 파일별 시간·주파수·엔벨로프 특징
  peaks.csv                          # 원신호/엔벨로프 주요 피크
  condition_summary.csv              # RPM·날짜·조건별 n, 평균/중앙값/P10/P90
  envelope_filter_response.csv       # 실제 사용 필터의 왕복 진폭 응답
  900rpm/20260912/                   # 나머지 RPM·날짜도 별도 폴더
    file_features.csv
    condition_summary.csv
    condition_spectra.png            # 저주파 확대 + 0–10kHz 비교
    feature_distributions.png        # 파일별 점 + 분포 box plot
    Normal1/                        # 각 조건 및 NormalCombined
      spectrum_distribution.csv     # bin별 평균/중앙값/P10/P90 + 평균 PSD
      envelope_distribution.csv     # 엔벨로프 평균/P10/P90
      waveform_spectra.png          # 대표 파형 + 스펙트럼 분포 + 엔벨로프
```

NormalCombined는 **같은 RPM·날짜의 Normal1·Normal2 파일을 파일 가중치로 통합**한다. 구성별 파일 수를 표시하므로 양쪽의 수가 다르면 동일 조건 가중 평균이 아님을 알 수 있다. 한 정상 조건만 있으면 해당 조건만 포함되고 나머지 수는 0이다. NormalCombined는 기존 Normal1/2와 같은 파일을 재사용하므로 독립 표본이 아니다. Bearing·Looseness·Unbalance의 각 세부 조건은 임의로 합치지 않는다.

대표 파형은 분석에 성공한 해당 그룹 파일 중 **AC RMS 중앙 순위 파일**이다(짝수 개면 아래쪽, 동률이면 경로순). 판별력이 높은 파일을 고르지 않는다. 대표 파일 경로는 CSV와 JSON에 기록된다. 스펙트럼의 P10–P90은 선택한 파일들의 기술적 분포이며 신뢰구간이 아니다. n=1이면 범위가 한 선으로 겹친다.

종료 코드 0은 분석 파일과 보고서 생성 완료, 2는 유효 측정 파일이 없어 분석 미수행, 1은 설정·출력·실행 오류다. 일부 파일이 제외되어도 나머지 분석은 가능하므로 `counts`, `counts_by_rpm`, 개별 제외 사유를 반드시 확인한다. 실패한 실행도 생성된 `analysis.json`의 `completed/status/error`를 확인한다. 기존 출력 디렉터리는 재사용하지 않는다.

## 해석 원칙

- 다른 RPM이나 날짜를 합친 분포를 결함의 차이로 해석하지 않는다. 동일 RPM·날짜의 정상 대조군이 없는 조건은 직접적인 정상/결함 차이를 결론내릴 수 없다.
- 날짜별 재현성을 먼저 살핀다. 같은 날의 변화도 부하·센서 부착·측정 순서 등의 영향일 수 있다.
- 원본 경로에 Run 독립성을 입증하는 정보가 없어 **독립성 미확인 탐색 분석**이다. 연속 파일은 같은 Run일 수 있으며, 여러 파일을 서로 독립된 반복 실험으로 간주하지 않는다.
- 특징 차이가 없으면 그 특징을 유효한 판단 근거라고 주장하지 않는다. 차이가 있어도 검증 전에는 결함 진단 규칙으로 전환하지 않는다. 이 도구 자체는 유효 특징을 자동 선정하거나 유의성·분류 성능을 산출하지 않는다.
- 후속 규칙 개발 시 독립 Run/날짜 분리, 정상 오경보율·검출률 평가, 센서/부하 교란 검토가 별도로 필요하다.

## 자동 테스트

```bash
MPLCONFIGDIR=/tmp/fault-signature-mpl \
  "$HOME/capstone-web-venv/bin/python" -B -m unittest tests.test_fault_signatures -v
```

정확한 FFT bin의 합성 정현파로 RMS·Peak/Crest 관계·진폭·1X/2X·해상도를 확인한다. 여러 대역 정현파의 PSD 평균제곱, 진폭변조 carrier의 엔벨로프 주파수 복원, 0 신호, NaN/Inf/overflow, 잘못된 필터·탐색 범위, 정확한 조건/날짜 파싱, 결정적 샘플 선정, 전체 6채널 검증, 원본 해시 보존, DB 미접근, CSV/PNG/JSON 생성 및 출력 경로 보호를 검사한다.

이번 클라우드 합성 테스트에서 실제 조건별 관측 결과는 생성하지 않는다. WSL에서 실행한 `analysis.json`과 CSV·PNG를 검토한 후에만 실제 관찰 사실을 기술할 수 있다.

2026-11-13 클라우드 실행: Python 3.14.4 / NumPy 2.4.6 / SciPy 1.18.1 / Matplotlib 3.11.2에서 신규 11개, 기존 V2 6개, AE 경로 일치 6개로 총 **23개 테스트가 통과**했다. CSV·PNG·JSON은 임시 폴더의 합성 데이터로 생성·검사한 뒤 테스트 종료 시 삭제했다. 실제 측정 데이터 결과 파일은 포함하지 않는다.
