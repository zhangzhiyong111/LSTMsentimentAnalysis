#encoding="utf-8"
#/usr/bin/env python

import sys
import unicodedata
import re
import codecs
import unicodedata
import urllib
from bs4 import BeautifulSoup

reload( sys )
sys.setdefaultencoding( "utf-8" )

def getPage( html ) :
	try :
		page = urllib.urlopen( html )
		page = page.read()
		pagecontent = page
	except :
		print "decode fail"
		return None
	return pagecontent

def parsedocBaidu( baiduweb ) :
	pagecontent = getPage( baiduweb )
	if pagecontent :
		soup = BeautifulSoup( pagecontent )
		content = soup.find( "div" , "content" )

		if content :
			# fw = codecs.open("./crawlGuangzhou_doc/%d_%d"%( maincount , count ),'a','utf-8')
			textContent = content.find_all( 'p' )
			if textContent :
				for item in textContent :
					Nonsense = item.find_all( 'a' )
					for sub_item in Nonsense :
						sub_item.decompose()
					paragraph = item.get_text( "" , strip = True )
					if len( paragraph ) < 8 :
						continue
					print paragraph
					print "..... "
					
			# 		fw.write( paragraph.encode( "utf-8" ) + "\n" )
			# fw.close()

def parsedocument( travelAddress , maincount , count  ) :
	pagecontent = getPage( travelAddress )
	if pagecontent :
		soup = BeautifulSoup( pagecontent )
		content = soup.find( "div" , "vc_article" )

		if not content :
			content = soup.find( "div" , "a_con_text cont")
		if content :
			fw = codecs.open("./crawlGuangzhou_doc/%d_%d"%( maincount , count ),'a','utf-8')
			textContent = content.find_all( 'p' )
			if textContent :
				for item in textContent :
					Nonsense = item.find_all( 'a' )
					for sub_item in Nonsense :
						sub_item.decompose()
					paragraph = item.get_text( "" , strip = True )
					if len( paragraph ) < 8 :
						continue
					
					fw.write( paragraph.encode( "utf-8" ) + "\n" )
			fw.close()

					
					
	else :
		print "The %d number of page decode fail "%sub_count

def parseMain( html , count ) :
	pagecontent = getPage( html )
	sub_count = 0

	if pagecontent :
		soup = BeautifulSoup( pagecontent )
		content = soup.find("div","att-list").find_all("div","flt1")

		for travelInfor in content :
			travelAddress = travelInfor.a['href']
			print travelAddress
			parsedocument( travelAddress , count , sub_count  )
			sub_count += 1

	else :
		"The %d number mainPage decode fail "%count
			
	# return nextHtml

def main() :
	# ss = "http://www.mafengwo.cn/group/s.php?q=%E5%B9%BF%E5%B7%9E%E6%B8%B8%E8%AE%B0&p=0&t=info&kt=1"
	# html = "http://www.mafengwo.cn/group/s.php?q=%E5%B9%BF%E5%B7%9E%E6%B8%B8%E8%AE%B0&t=info&seid=4702EB64-957F-4D5D-9E04-E629E0F59905&mxid=0&mid=0&mname=&kt=1"
	# This is for mafengwo travelogues
	# count = 0
	# frontHtml = "http://www.mafengwo.cn/group/s.php?q=%E5%B9%BF%E5%B7%9E%E6%B8%B8%E8%AE%B0&p="
	# followHtml = "&t=info&kt=1"
	# for count in range( 2 , 50 ) :
	# 	parseMain( frontHtml + str( count + 1 ) + followHtml , count )

	baiduweb = "https://lvyou.baidu.com/notes/63f2fd27eb0ac336aec28e7d?request_id=2594101413&idx=0"
	parsedocBaidu( baiduweb )

if __name__ == '__main__' :
	main()