// ──────────────────────────────────────────────
// HR 챗봇 프론트엔드 서버 (Express + 프록시)
//
// 역할:
//   1) index.html 등 정적 파일을 브라우저에 서빙
//   2) 브라우저의 /api/* 요청을 백엔드(localhost:8000)로 중계 (프록시)
//
// 프록시를 두는 이유:
//   - 브라우저는 다른 출처(다른 포트)로 직접 요청 보내면 CORS 정책에 막힘
//   - 프론트 서버를 통해 백엔드로 전달하면 같은 출처에서 요청한 셈이 되어 차단 없음
//   - 백엔드 코드를 수정하지 않고도 CORS 문제 해결
//
// 예외 처리 원칙:
//   - 백엔드(FastAPI)가 죽어 있거나 네트워크 오류가 나도 이 프론트 서버는 죽지 않게 한다.
//   - 프록시 onError 핸들러로 502 JSON 응답을 돌려준다 (index.html 의 에러 처리에서 인식 가능).
//   - 라우트 처리에서 예외가 나면 글로벌 에러 미들웨어가 500 JSON 응답을 돌려준다.
//   - 잡히지 않은 비동기 에러(uncaughtException / unhandledRejection)는 콘솔에만 찍고
//     프로세스를 종료시키지 않는다. (Node 기본 동작이 종료라서 명시적으로 막아야 한다)
// ──────────────────────────────────────────────

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

// 같은 폴더(dma/)의 .env 파일을 읽어 환경변수로 등록
require('dotenv').config({ path: path.join(__dirname, '.env') });

const app = express();

// 프론트 서버가 사용할 포트 (.env의 FRONT_PORT, 기본 3000)
const PORT = process.env.FRONT_PORT || 3000;

// 백엔드(FastAPI) 주소 (.env의 BACKEND_URL, 기본 localhost:8000)
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// ──────────────────────────────────────────────
// 1. /api/* 요청을 백엔드로 중계
//    예: 브라우저가 POST /api/rag-chat 보내면
//        → 서버가 POST http://localhost:8000/rag-chat 로 전달
//
//    onError 핸들러:
//      - 백엔드가 다운됐거나 연결 오류면 createProxyMiddleware 가 이 함수를 호출한다.
//      - 응답이 이미 보내진 상태가 아니면 502 JSON 을 돌려준다.
//      - 이 핸들러가 없으면 Express 가 기본 에러 페이지(HTML)를 돌려주거나
//        프록시 라이브러리가 예외를 던져 프로세스가 죽을 수 있다.
// ──────────────────────────────────────────────
app.use(
  '/api',
  createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    // 요청 경로에서 '/api' 부분 제거하고 백엔드로 전달
    pathRewrite: { '^/api': '' },
    // v3 이후 proxy 이벤트 핸들러는 on: { error, proxyReq, ... } 형태로 등록한다.
    // (이전 'onError' 키는 v3 에서 무시되므로 반드시 on.error 로 둬야 한다.)
    on: {
      error: (err, req, res) => {
        console.error('[프록시 에러]', err && err.message ? err.message : err);

        // 응답을 아직 못 쓴 상태일 때만 우리가 직접 응답을 만들어 보낸다.
        if (res && !res.headersSent) {
          res.status(502).json({
            detail: '백엔드 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.',
          });
        }
      },
    },
  })
);

// ──────────────────────────────────────────────
// 2. 정적 파일 서빙 (style.css, chatbot.js)
//    파일명을 명시적으로 지정해 CSS·JS만 노출
//    (CSV·백엔드 소스 등 민감 파일은 노출 안 됨)
// ──────────────────────────────────────────────
app.get('/style.css', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'style.css'));
});
app.get('/chatbot.js', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'chatbot.js'));
});

// ──────────────────────────────────────────────
// 3. index.html 서빙
//    파일 전송이 실패하면(권한 문제, 파일 삭제 등) next(err) 로 에러 미들웨어에 넘긴다.
// ──────────────────────────────────────────────
app.get('/', (req, res, next) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'), (err) => {
    if (err) {
      next(err);
    }
  });
});

// ──────────────────────────────────────────────
// 4. 글로벌 에러 미들웨어
//    Express 는 인자가 4개(err, req, res, next)인 미들웨어를 '에러 처리용'으로 인식한다.
//    위 라우트나 다른 미들웨어에서 next(err) 가 호출되면 여기로 모인다.
//    이 미들웨어가 없으면 Express 가 스택 트레이스를 그대로 응답으로 보내거나
//    경우에 따라 프로세스가 죽을 수 있다.
// ──────────────────────────────────────────────
app.use((err, req, res, next) => {
  console.error('[글로벌 에러 미들웨어]', err && err.stack ? err.stack : err);

  if (!res || res.headersSent) {
    return;
  }

  // 요청이 HTML 페이지를 기대하는 경우(브라우저 주소창 직접 진입 등)에는
  // JSON 대신 간단한 HTML 페이지를 돌려준다. 그래야 사용자가 raw {"detail":...} 가 아닌
  // 의미 있는 화면을 보게 된다.
  if (req.accepts(['html', 'json']) === 'html') {
    res.status(500).type('html').send(
      '<!doctype html><meta charset="utf-8"><title>오류</title>' +
      '<div style="font-family:sans-serif;max-width:480px;margin:80px auto;text-align:center">' +
      '<h1 style="color:#991B1B">서버 내부 오류가 발생했습니다.</h1>' +
      '<p>잠시 후 다시 시도해주세요.</p>' +
      '</div>'
    );
    return;
  }

  // 그 외(API 호출 등)에는 기존처럼 JSON 으로 응답한다.
  res.status(500).json({
    detail: '서버 내부 오류가 발생했습니다.',
  });
});

// ──────────────────────────────────────────────
// 5. 프로세스 단위 안전망
//    Node.js 공식 권장: uncaughtException 이후 프로세스 상태는 신뢰할 수 없으므로
//    로그를 남기고 반드시 종료해야 한다. PM2/systemd 같은 프로세스 매니저가 재시작을 책임진다.
//    여기서 살려두면 후속 요청이 손상된 상태로 처리될 수 있다.
// ──────────────────────────────────────────────
process.on('uncaughtException', (err) => {
  console.error('[uncaughtException]', err && err.stack ? err.stack : err);
  // 비동기 로그가 flush 될 시간을 짧게 주고 종료한다.
  setTimeout(() => process.exit(1), 100);
});

process.on('unhandledRejection', (reason) => {
  console.error('[unhandledRejection]', reason);
  setTimeout(() => process.exit(1), 100);
});

// ──────────────────────────────────────────────
// 6. 서버 시작
// ──────────────────────────────────────────────
app.listen(PORT, () => {
  console.log('────────────────────────────────────────');
  console.log(`HR 챗봇 프론트 서버 실행 중`);
  console.log(`프론트 주소: http://localhost:${PORT}`);
  console.log(`백엔드 연결 대상: ${BACKEND_URL}`);
  console.log('────────────────────────────────────────');
});
