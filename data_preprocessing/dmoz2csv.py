class RDF:

    def __init__(self, filename=''):
        self.in_filename = filename
        self.fd = open(filename,'r', encoding="utf8")
        self.urls_serial = 0
        self.read_urls = []
        self.topics = {}

    def __enter__(self):
        #print 'Enter RDF'
        return self

    def __exit__(self, type, value, traceback):
        #print 'Exit RDF'
        self.fd.close()

    def add_url(self, url='', topic='', title='', desc='', lower=True):
        # To turn Kids_and_Teens into Kids
        topic = topic.split('_')[0]
        url = url.replace(',','')
        if lower:
            url = url.lower()
            title = title.lower() if title else title
            desc = desc.lower() if desc else desc
        # Baykan-2009 ignores those topics
        ignored_topics = ['World', 'Regional']
        if not topic in ignored_topics:
            self.urls_serial += 1
            self.read_urls.append([url, topic, title, desc])
#             self.read_urls.append([self.urls_serial, url, topic])
            if topic in self.topics:
                self.topics[topic] += 1
            else:
                self.topics[topic] = 1 

    def showTopics(self):
        for topic in self.topics:
            print(topic, ':', str(self.topics[topic]))

    def getPageURL(self, line=''):
        return line.split('"')[1] 

    def getPageTopic(self):
        topic = ''
        #for line in self.fd.readlines():
        for line in self.fd:
            line = line.strip()
            #print 'getPageTopic:', line
            if line.startswith('<topic'):
                #print line
#                 topic = line.rsplit('<',1)[0].split('/')[1]
                topic = line.rsplit('<',1)[0].replace('<topic>', '')
            elif line.startswith('</ExternalPage'):
                return topic  
    
    def getTitle(self):
        title = ''
        for line in self.fd:
            line = line.strip()
            if line.startswith('<d:Title'):
                title = line.rsplit('<',1)[0].replace('<d:Title>', '').replace(',', '')
            elif line.startswith('</ExternalPage'):
                return title
    
    def getDescription(self):
        desc = ''
        for line in self.fd:
            line = line.strip()
            if line.startswith('<d:Description'):
                desc = line.rsplit('<',1)[0].replace('<d:Description>', '').replace(',', '')
            elif line.startswith('</ExternalPage'):
                return desc
    
    def getPages(self):
        #for line in self.fd.readlines():
        for line in self.fd:
            line = line.strip()
            if line.startswith('<ExternalPage'):
                url = self.getPageURL(line)
                topic = self.getPageTopic()
                title = self.getTitle()
                desc = self.getDescription()
                self.add_url(url, topic, title, desc)
        print('Read URLs:', len(self.read_urls), '\n')

    def writeCSV(self):
        csv_finename = self.in_filename.split('.')[0] + '.csv'
        fd = open(csv_finename, 'w', encoding="utf8")
        for u in self.read_urls:
            line = '%s,%s,%s,%s\n' % tuple(u)
            fd.write(line)
        fd.close()

def main(filename=''):

    with RDF(filename) as rdf:
        rdf.getPages()
#         rdf.showTopics()
        rdf.writeCSV()

if __name__ == '__main__':

    main(filename='I:/Desktop/DMOZ/content.rdf.u8')