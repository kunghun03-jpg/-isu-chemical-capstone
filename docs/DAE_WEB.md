# Streamlit DAE 통합

기존 AE·OCSVM·IF의 추론 API, 결과 JSON, 8구간 상세 화면과 이력 DB는 유지합니다.
모델 선택에 DAE를 추가하고 DAE 결과는 별도 세션 값과 화면으로 처리합니다.
`전체`는 기존 세 모델 실행 후 DAE가 사용 가능할 때만 DAE를 별도로 실행합니다.
DAE 누락·RPM 차단·추론 실패는 기존 세 모델 결과를 지우지 않습니다.

## 준비 상태

| 표시 | 동작 |
| --- | --- |
| 모델 미준비 | 번들/필수 파일 없음. DAE 단독 분석 비활성화 |
| RPM 적용 범위 미확인 | supported_rpms=null. DAE 판정 차단 |
| 선택 RPM 미지원 | 확인된 목록에 현재 RPM 없음. DAE 판정 차단 |
| 모델 검증 실패 | manifest·버전·해시 검증 오류. DAE 판정 차단 |
| 사용 가능 | 선택 RPM과 manifest 검증 통과. 실행 시 실제 통계·가중치와 RPM을 다시 검사 |

DAE 단독 미리보기와 추론은 DAE 전용 Welch 전처리를 사용합니다.
기존 세 모델 미리보기/8구간 FFT 전처리를 DAE에 적용하지 않습니다.
모델·업로드·RPM 변경 시 결과를 지웁니다. 번들 누락·해시/준비 상태 변경 시에도
기존 DAE 결과를 지우며, 실행 중 번들 식별자가 변경되면 분석을 중단합니다.

## 전용 결과 화면

- 파일 전체 MSE·저장 threshold·정상/이상 판정
- 800특성 입력/복원 스펙트럼 (Welch 제곱근·pooling·log10·저장 Z-score 적용값)
- 기존 세 모델의 8구간 그래프·구간 threshold는 DAE에 생성하지 않음
- DAE는 DB에 저장하지 않음. `전체`의 기존 저장/JSON 버튼도 기존 세 모델 결과만 포함

## WSL 번들 연결

기본 경로는 프로젝트의 `outputs/dae_bundle/`입니다. 실제 파일은 체크아웃 밖에 두고
환경변수로 지정하는 방법을 권장합니다. 예를 들어 사용자가 보고한 검증 산출물은 다음과 같이 연결합니다.

```bash
ISU_CHEMICAL_DAE_BUNDLE="$HOME/dae-validation/20261113T041136515404Z/bundle" python -m streamlit run streamlit_app.py
```

이 경로의 실제 파일은 클라우드에서 확인하지 않았습니다. 전달된 보고에 따르면 기존 가중치와
1200RPM IR MAT 한 개에서 `compatibility_checks_passed_rpm_unconfirmed; input=provided_mat`
결과를 얻었습니다. 이는 사용자 제공 계산 호환성 결과이며 독립적으로 보고서를 열람/재검증한
기록이나 탐지 성능·1200RPM 지원 확정이 아닙니다. 해당 번들은 그대로 RPM 미확인 상태를
표시하고 웹 판정을 차단해야 합니다.

웹에서 실제 DAE 판정을 활성화하려면 학습/분할/정규화 기록과 RPM별 검증 근거를 확보한 후,
확인된 supported_rpms 및 rpm_evidence를 지정한 유효 번들을 별도로 생성해야 합니다.
기존 manifest를 임의 편집하거나 미확인 차단을 우회하지 않습니다. 실행 환경의 NumPy/SciPy/
PyTorch 버전도 번들과 일치해야 하며, 최종적으로 실제 MAT로 웹 결과를 legacy/CLI와 대조해야 합니다.

## 합성 검사

`python -m unittest tests.test_dae_ui -v`

테스트의 [900] RPM 선언은 임시 가중치의 UI 테스트용 설정이며 실제 모델의 지원 범위를
뜻하지 않습니다. 합성 DAE 파일로 UI/API 계산 일치와 차단·오류·상태 초기화를 검사하고,
기존 세 모델은 테스트 대역으로 결과 전달·화면·저장 경로 보존을 확인합니다.
실제 세 모델 및 DAE의 성능을 이 UI 테스트로 검증한 것은 아닙니다.
