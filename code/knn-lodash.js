const outputs = [
			[ 10, .5, 16, 1],
			[200, .5, 16, 4],
			[350, .5, 16, 4],
			[600, .5, 16 , 5]
			];

const k = 3;

function distance(pointA, pointB) {
	return Math.abs(pointA - pointB);
}

function getTrainTestSets(data,testSetSize){
	const shuffled = _.shuffle(data);
	
	const testSet = _.slice(shuffled,0,testSetSize);
	const trainingSet = _slice(shuffled,testCount);
	
	return [testSet, trainingSet];
}

function knn(data, point){
	return _.chain(data)
				.map(row => [distance(row[0],point),row[3]])
				.sortBy(row => row[0])
				.slice(0,k)
				.countBy(row => row[1]
				.toPairs()
				.sortBy(row => row[1]
				.last()
				.first()
				.parseInt(),
				.value();
}

function getAccuracy(trainingSet, testSet){
	let numberCorrect = 0;
	const testSetSize = testSet.length;
	for(let i = 0; i < testSetSize; i++){
		const prediction = knn(trainingSet, testSet[i][0]);
		if(prediction === testSet[i][3]){
			numberCorrect++;
		}
	}
	return numberCorrect / testSetSize;	
}