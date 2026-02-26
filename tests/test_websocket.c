/* Test WebSocket API and linkage */
#include "websocket.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    printf("test_websocket: ");
    assert(WS_TEXT == 0x1 && WS_BINARY == 0x2 && WS_CLOSE == 0x8);
    assert(WS_PING == 0x9 && WS_PONG == 0xA);
    printf("OK\n");
    return 0;
}
