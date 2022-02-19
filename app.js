// const { reviews } = require('./data/naver_comments.json');
// // TODO: case1. <br> 엔터가 없는 케이스
// let idx = 1
// review = reviews[idx] // '배송이 빨라요. 3일에 주문해서 5일에 도착했어요.참고로 지방에 거주합니다. 아이패드 배송 요새 많이 기다리는 것 같았는데 이렇게도 오네요. 따로 주문한 아이패드 악세서리보다 일찍 왔어요. <em>제품자체는 양품테스트 해봤는데 다 괜찮다가</em> 저녁되어서 방금 해본 빛샘테스트에서 한 구석탱이가 긴가민가도 아닌 딱 확연해서 아쉬웠습니다..만 애케플도 있고 그냥 일반적으로 사용하기에 문제도 없을 것 같고 다른부분이 양호해서 안고가려고 합니다... ^_ㅠ 이 정도면 양품인 것 같아요. 그리고 스그 예쁩니다 고민했는데 <em>예뻐서 좋아요</em> ㅎ ㅎ';
// let {content, topics} = review;

// did_replace = content.replace(/\<br\>/gi, ' ').replace(/(<([^>]+)>)/ig, '')
// console.log('**before**')
// console.log(content)
// console.log('**after**')
// console.log(did_replace);
// buffer = Buffer.from(did_replace, 'utf-8');

// console.log()
// console.log(buffer.slice(topics[0].startPosition, topics[0].endPosition-2).toString())
// console.log(buffer.slice(topics[1].startPosition, topics[1].endPosition-2).toString())

// // // TODO: <br> 엔터가 하나 있는 케이스
origin_str = '저녁에 주문했는데 바로 다음날 발송처리 되었고 그 다음날 도착했습니다.<br>진짜 아이패드는 뽑기운도 따라줘야 하잖아요ㅠㅠ 여기는 액정기스나 찍힘은 교환사유가 되지 않는다고 해서 걱정많았는데 다행히도 양품 받았습니다. 유튜브에서 불량테스트 모두 해서 통과하였고 <em>너무너무 예뻐요</em>ㅎㅎㅎ 몇달동안 살까말까 고민 많았는데 <em>진짜 너무 만족합니다</em>. 공부용으로 산건데 공부하고 싶은 의욕이 막 생겨요ㅋㅎ 그리고 색감은 최대한 비슷하게 담으려고 노력했는데, 스카이블루 보다는 <em>스카이 블루에 그레이나 실버색이 섞인 느낌이고 더 고급</em>져보이고 영롱해용! 2만원 더 주고 스블 산 <em>보람이 있네요</em>ㅎ.ㅎ';
did_replace = origin_str.replace(/\<br\>/gi, ' ').replace(/(<([^>]+)>)/ig, '')
console.log('**before**')
console.log(origin_str);
console.log('**after**')
console.log(did_replace);
buffer = Buffer.from(did_replace, 'utf-8');

console.log()
console.log(buffer.slice(368-3, 389-2).toString())
console.log(buffer.slice(446-3, 474-2).toString())
console.log(buffer.slice(652-3, 727-2).toString())
console.log(buffer.slice(705-3, 727-2).toString())
console.log(buffer.slice(785-3, 803-2).toString())

// TODO: <br> 엔터가 하나도 없는 케이스
origin_str = '저는 아이패드 에어4 로즈골드 64GB를 구매했습니다[배송]10/27에 사전예약으로 11/6일 발송이었지만 11/3에 배송을 받을 수있었습니다.아이패드 에어4가 인기가 좋고 사람들이 많이들 찾으셔서 배송이 늦을 수도 있다고 생각했는데 생각보다는 빨랐습니다.[포장]뽁뽁이 2겹으로 해주셔서 걱정하시는 분들은 걱정안하셔도 될것 같습니다.택배박스를 다른 택배박스보다 튼튼한 느낌입니다.(물론 뜯을때도 힘들었습니당 ㅎㅎㅎ)[<em>디자인]저는 아이패드를 처음받자마자 이쁘다라는</em> 생각밖에 들지 않았습니다.<em>로즈골드는 우아한 아름다움</em>?같은 느낌이네용.[<em>성능]정말 너무 좋습니다</em>.저는 넷플릭스 유튜브 공부정도로 사용하는데 너무 충분해서 차고 넘칩니다.실사용에서 흠잡을 곳이 없네용.[총평]저도 아이패드 프로와 고민을 했지만 저처럼 영상시청 공부와 그림위주의 사용이시라면 <em>강력하게 추천</em> 드리고 싶어용.“고민은 배송만 늦출뿐입니당”';
did_replace = origin_str.replace(/\<br\>/gi, ' ').replace(/(<([^>]+)>)/ig, '')
console.log('**before**')
console.log(origin_str);
console.log('**after**')
console.log(did_replace);
buffer = Buffer.from(did_replace, 'utf-8');

console.log()
console.log(buffer.slice(559, 625-2).toString())
console.log(buffer.slice(663, 700-2).toString())
console.log(buffer.slice(726, 758-2).toString())
console.log(buffer.slice(1031, 1049+2).toString())

