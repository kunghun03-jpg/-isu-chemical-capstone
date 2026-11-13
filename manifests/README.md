# Run 목록 관리

- `run_inventory_unconfirmed.csv`: 현재 원본 파일 목록. Run ID와 split은 미확정 빈칸입니다.
- `confirmed_*.csv`: 실험 기록을 확인한 뒤 작성할 확정 목록입니다. 아직 만들지 않았습니다.

필수 열: `file,date,rpm,condition,run_id,split`.
추가 집계 열: `source_condition,is_normal,default_included`.
정상 상태는 Normal1·Normal2·Normal3이며 Normal3는 default_included=0으로 기록하고
기본 실험 split은 비워 둡니다. 결함 상태의 Looeness는 condition에서 Looseness로 통일합니다.
실제 경로와 source_condition은 원본 표기를 유지합니다.
같은 실제 Run은 전체 목록에서 같은 ID를 사용하고 하나의 split에만 배정합니다.
원본 목록을 보존하고 복사본에서 확정 정보를 관리하세요.
통합 목록은 3개 날짜·3개 RPM의 전체 4,871개 파일로 갱신했습니다.
Run ID와 split은 Normal3를 포함하여 모두 미확정 빈칸이며 자동 확정하지 않았습니다.
