def get_visual_tags(entry):


    tags = ''

    if 'image_type' in entry:
        for i in entry['image_type']:
            if i['score'] >= 0.8:
                tags += i['label'] + ' with '




    if 'face' in entry:

        face_count = 0
        emotions = set()

        for i in entry['face']:

            if i['detection']['score'] >= 0.9:
                face_count += 1
                max_emotion = ''
                max_score = 0

                for emotion in i['emotion']:
                    if emotion['score'] > max_score:
                        max_emotion = emotion['label']
                        max_score = emotion['score']

                if max_score >= 0.5:
                    emotions.add(max_emotion)

        if face_count > 0:
            if face_count == 1:
                tags += str(face_count) + ' person, '
            else:
                tags += str(face_count) + ' people, '

            if len(emotions) > 0:
                tags += ' with facial expression: '
                for e in emotions:
                    tags += ' '+e+' '

    if 'object_detection' in entry:
        objects = set()

        for o in entry['object_detection']:
            if o['score'] >= 0.9:
                if o['label'] == 'person' and face_count > 0:
                    continue
                else:
                    objects.add(o['label'])

        if 'person' in objects:
            tags += ' person, '
            objects.remove('person')

        if len(objects) > 0:
            tags += ' '
            for o in objects:
                tags += o.replace('_', ' ')+', '

    scenes = ''
    if 'indoor_scene' in entry:
        for i in entry['indoor_scene']:
            if i['score'] >= 0.8:
                scenes += i['label'].replace('_', ' ').replace('/', ', ') + ', '

    if 'places' in entry:

        scenes = ''
        for i in entry['places']:
            if i['score'] >= 0.8:
                scenes += i['label'].replace('_', ' ').replace('/', ', ') + ', '

    if scenes != '':
        tags += ' ' + scenes
    return tags.strip()
