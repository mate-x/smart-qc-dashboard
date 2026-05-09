# Claude Code 프롬프트 템플릿

> 조회표 (태스크명·참조 PRD): [docs/TEAM_ROLES.md §7](./TEAM_ROLES.md#7-claude-code-프롬프트-가이드)

## 구현 프롬프트

```
@docs/PRD/00_Global_Context_Document.md

나는 비전검사 대시보드 프로젝트의 [팀원 A / 팀원 B / 팀원 C]이야.

지금 [태스크명]을 진행해줘.

작업 시작 전 app.py, utils/, tabs/, components/, tests/ 의 현재 파일 목록을 확인하고,
이미 존재하는 파일은 새로 생성하지 말고 필요한 경우 수정만 해.

- 참조 문서
@docs/PRD/[XX_파일명.md]
```
