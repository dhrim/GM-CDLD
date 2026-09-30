# 자료 명세

NBA 원자료는 shufinskiy/nba_data의 commit e829d4678be1e075f99e5d41a1c5f97089be446b에서 확보한 pbpstats_2023·cdnnba_2023 및 공식 boxscore 자료를 사용했다. 테니스 2017–2019 파일 URL와 획득 hash는 provenance/tennis_sources.json에 있다.

학습 입력은 도메인별 fit.npz/test.npz/meta.json/player_ids.npy이다. npz 필드 a,b는 각 팀 선수의 latent 행 index, context는 발견망 맥락, row_id/cluster는 분할 식별자이며 정답 이름은 protocol/plan.json의 targets를 따른다. 테니스 재사용에는 ace_rate_other/df_rate_other/service_win_rate_other도 필요하다.

본 패키지에 준비된 정답 배열은 없다. 해당 준비 자료를 정당하게 확보한 경우 tools/stage_run.py --data 경로 --output 새폴더로 반입한다. 메타데이터와 분할 식별자는 동봉했다.

provenance/prepare_data_original.py는 실제 전처리 코드 기록이다. 이전 NBA 전처리 산출물에 의존하므로 독립 실행용 스크립트로 제시하지 않는다. 필요한 입력은 코드의 read_csv/read_parquet/tarfile 경로에 명시돼 있다. 연구자의 원본 작업 기록에 준비 배열과 그 상위 산출물이 보존돼 있다.


데이터 저작권·이용 조건은 [RIGHTS.md](../RIGHTS.md)의 제공처별 조건을 따른다. 테니스 CC BY-NC-SA 4.0과 농구 저장소 Apache 2.0 표기를 구분하며 원제공자 권리도 별도로 고려한다.


## Current notebook support

Tennis: notebook 01 downloads the three upstream files, verifies the archived SHA256 hashes, and runs the unchanged tennis transformation extracted from the original preparation script. Its output is work/tennis_prepared/data/tennis. Copy this directory under the main prepared input root alongside NBA/.

NBA: supply NBA/fit.npz, test.npz, meta.json and player_ids.npy under that root. The archived preparation script requires the earlier lineup/target pipeline outputs: schedule_and_candidate_split.csv, cdn_raw.parquet, player_boxscores_2024.parquet, pbpstats_2023.tar.xz, and the two target_rows.parquet sources shown in that script. These are not included. It is not a standalone raw-data builder. Completing that upstream packaging remains a release task.

Beach volleyball: supply source/ and reuse/, each containing fit.npz, val.npz and meta.json. Test files are not staged by notebook 04. The archived preparation chain depends on previous split construction; it remains a release task. See extensions/beach_volleyball/source_manifest.json for upstream sources.

The published data/ folder contains only split identifiers, player IDs and metadata, not outcome arrays. Do not rename a later C74 or shot-success dataset to stand in for the manuscript NBA points data.
