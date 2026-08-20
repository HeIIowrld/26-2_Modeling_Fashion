/* 백엔드 API 주소.
 *
 * 빈 문자열이면 이 페이지를 내려준 서버로 그대로 요청한다. `python web/run_web.py`처럼
 * FastAPI가 화면까지 함께 서빙하는 경우가 여기에 해당한다.
 *
 * GitHub Pages 같은 정적 호스팅에 올릴 때는 화면만 있고 API가 없으므로,
 * 연산을 맡을 서버 주소를 넣어야 한다. 배포 워크플로(.github/workflows/static.yml)가
 * 저장소 변수 FASHION_API_BASE 값으로 이 파일을 덮어쓴다.
 *
 *   예) window.FASHION_API_BASE = "https://fashion-api.example.com";
 */
window.FASHION_API_BASE = "";
