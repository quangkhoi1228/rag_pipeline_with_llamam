import http from 'k6/http';

export const options = {
    scenarios: {
        ramptest: {
            executor: 'constant-arrival-rate',
            duration: '30s',
            preAllocatedVUs: 10,
            rate: 120
        }
    }
}

export default function () {
    const url = "http://localhost:8000/chat/send/?version=14-pro"
    
    const payload = JSON.stringify({
        "message":"Luật về đất đai có ảnh hưởng gì tới người dân",
        "history_count":6
    })

    const params = {
        headers: {
          'Content-Type': 'application/json',
        },
      };
  http.post(url, payload, params);
}